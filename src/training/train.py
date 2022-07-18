import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

#from open_clip import ClipLoss
from training.loss import gather_features, ClipLoss, ProtoLoss
from .distributed import is_master, get_gathered_item
import torch.distributed as dist
from training.evaluations.analyze_features import get_modality_gap
from utils.training_utils import AverageMeter, unwrap_model


def train_one_epoch(
    student, text_teacher, image_teacher, 
    data, epoch, optimizer, scaler, scheduler, 
    distiller_text, distiller_image, args, tb_writer,
    adaption_head=None, adaption_head_optimizer=None, panalty_scheduler=None
    ):
    device = torch.device(args.device) 
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    student.train()
    if text_teacher is not None:
        text_teacher.eval()
    if image_teacher is not None:
        image_teacher.eval()
    if args.adaption_head:
        adaption_head.train()
    
    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    if is_master(args):
        logging.info(f'Using [{args.distiller}] distiller')
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        index, images, texts = batch
        if len(index)!=args.batch_size:
            continue  # drop last incomplete small batch
        
        #all_index = get_gathered_item(index.cuda(), args)
        images = images.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.adaption_head:
            adaption_head_optimizer.zero_grad()

        with autocast():
            # # # # # # # # # # # # # # # # # # 
            # Text Teacher forward
            # # # # # # # # # # # # # # # # # # 
            if text_teacher is not None:
                with torch.no_grad():
                    text_features_t = text_teacher.encode(texts, convert_to_tensor=True,  show_progress_bar=False)
                if args.adaption_head:
                    adaption_shift = adaption_head(text_features_t)
                    text_features_t += adaption_shift
                with torch.no_grad():# FIXME: possible bug due to stoped gradient?
                    if args.distiller in ['ProtoCPC', 'DINO']:
                        prototype = student.module.text_projection_head.last_layer if args.distributed else student.text_projection_head.last_layer
                        text_features_t = prototype(text_features_t)
            
            # # # # # # # # # # # # # # # # # # 
            # Image Teacher forward
            # # # # # # # # # # # # # # # # # # 
            if image_teacher is not None:
                with torch.no_grad():
                    image_features_t = image_teacher(images)
                    if args.distiller in ['ProtoCPC', 'DINO']:
                        prototype = student.module.image_projection_head.last_layer if args.distributed else student.image_projection_head.last_layer
                        image_features_t = prototype(image_features_t)

            # # # # # # # # # # # # # # # # # # 
            # Image Student forward
            # # # # # # # # # # # # # # # # # # 
            if args.freeze_student_backbone:
                with torch.no_grad():
                    student_features = student(images).detach()
            else:
                student_features = student(images)
            
            if text_teacher is not None:
                student_features_text = student.module.text_projection_head(student_features) if args.distributed else student.text_projection_head(student_features)
                text_loss = distiller_text(text_features_t, student_features_text)
                if args.adaption_head:
                    panalty_weight = panalty_scheduler[step]
                    adaption_norm = torch.norm(adaption_shift, p=2, dim=0).mean()
                    text_loss += panalty_weight * adaption_norm
            else:
                text_loss = 0

            if image_teacher is not None:
                student_features_image = student.module.image_projection_head(student_features) if args.distributed else student.image_projection_head(student_features)  
                image_loss = distiller_image(image_features_t, student_features_image)
            else:
                image_loss = 0
            
            total_loss = text_loss + image_loss
            
            # # # # # # # # # # # # # # # # # # 
            # loss backward
            # # # # # # # # # # # # # # # # # # 
            # FIXME: gather skiped. InfoNCE will degenerate
            # if args.distributed:
            #     all_image_features, all_text_features = gather_features(
            #         image_features, text_features,
            #         args.local_loss, args.gather_with_grad, 
            #         args.rank, args.world_size, args.horovod
            #         )
            # else:
            #     all_image_features, all_text_features = image_features, text_features
            

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.max_grad_norm)
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

        if args.adaption_head:
            adaption_head_optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            logging.info(
                f"Train Epoch: {epoch+1}/{args.epochs} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {total_loss.item():.5f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:3f} "
                f"grad: {norm:1f} "
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss-total": total_loss.item(),
                "loss-image": image_loss.item() if image_teacher is not None else 0,
                "loss-text": text_loss.item() if text_teacher is not None else 0,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient-norm": norm,
            }
            profiling = {
                "batch data time (s)": data_time_m.val,
                "bathc total time (s)": batch_time_m.val,
            }
            
            if args.adaption_head:
                log_data['adaption-shift-norm'] = adaption_norm
                log_data['adaption-panalty-weight'] = panalty_weight

            for name, val in log_data.items():
                name = "training/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            for name, val in profiling.items():
                name = "profiling/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


