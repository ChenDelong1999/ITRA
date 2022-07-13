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

MSE = nn.MSELoss()

def train_one_epoch(student, teacher, data, epoch, optimizer, scaler, scheduler, distiller, args, tb_writer):
    device = torch.device(args.device) 
    ZERO = torch.zeros(1).to(args.device)
    
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    student.train()
    teacher.eval()
    
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
        if len(index)!=args.batch_size: # drop last incomplete small batch
            continue
        all_index = get_gathered_item(index.cuda(), args)
        images = images.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        total_loss = 0
        with autocast():
            # Text Teacher
            with torch.no_grad():
                raw_text_features = teacher.encode(
                    texts,
                    convert_to_tensor=True, 
                    show_progress_bar=False
                    ).detach()
            if args.add_teacher_projection_head:
                if args.distributed:
                    text_features = student.module.text_projection_head(raw_text_features) + raw_text_features if args.res_teacher_projection else student.module.text_projection_head(raw_text_features)
                    if args.add_teacher_projection_AE:
                        reconstructed_text_features = student.module.text_decoder(text_features) + text_features if args.res_teacher_projection else student.module.text_decoder(text_features)
                        loss_projection_AE = MSE(reconstructed_text_features, raw_text_features)
                        total_loss += args.w_projection_AE * loss_projection_AE
                else:
                    text_features = student.text_projection_head(raw_text_features) + raw_text_features if args.res_teacher_projection else student.text_projection_head(raw_text_features)
                    if args.add_teacher_projection_AE:
                        reconstructed_text_features = student.text_decoder(text_features) + text_features if args.res_teacher_projection else student.text_decoder(text_features)
                        loss_projection_AE = MSE(reconstructed_text_features, raw_text_features)
                        total_loss += args.w_projection_AE * loss_projection_AE
            else:
                text_features = raw_text_features

            # Image Student
            if args.freeze_student_backbone:
                with torch.no_grad():
                    raw_image_features = student(images).detach()
            else:
                raw_image_features = student(images)
                
            if args.distributed:
                image_features = student.module.image_projection_head(raw_image_features)
            else:
                image_features = student.image_projection_head(raw_image_features)

            if args.distributed:
                all_image_features, all_text_features = gather_features(image_features, text_features,
                    args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
            else:
                all_image_features, all_text_features = image_features, text_features
            
            total_loss += distiller(all_text_features, all_image_features)

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
            feature_std_image = torch.std(F.normalize(all_image_features, dim=1), dim=0).mean().item()
            feature_std_text = torch.std(F.normalize(all_text_features, dim=1), dim=0).mean().item()
            log_data = {
                "total_loss": total_loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient-norm": norm,
                "feature_std_ratio": feature_std_image/feature_std_text,
                "feature_std_image": feature_std_image,
                "feature_std_text": feature_std_text,
                #"feature_modality_gap": get_modality_gap(image_features, text_features),
            }
            profiling = {
                "batch data time (s)": data_time_m.val,
                "bathc total time (s)": batch_time_m.val,
            }

            if args.add_teacher_projection_AE:
                log_data['loss_projection_AE'] = loss_projection_AE.item()

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

# TODO: feature extraction for teacher-student is not implemented yet
def feature_extraction_one_epoch(model, data, epoch, optimizer, scaler, scheduler, clustering, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.eval()

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):

        indexs, images, texts = batch
        indexs = indexs.to(device=device, non_blocking=True)
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)

        # forward propagation
        with autocast():
            with torch.no_grad():
                image_features, text_features, image_features_projected, text_features_projected, logit_scale, logit_scale_proto = model(images, texts)
        
        # cache features
        indexs = get_gathered_item(indexs, args)
        image_features_projected = get_gathered_item(image_features_projected, args)
        text_features_projected = get_gathered_item(text_features_projected, args)
        if is_master(args):
            clustering.load_batch(indexs, image_features_projected, text_features_projected)
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            logging.info(
                f"Feature extraction: {epoch+1}/{args.epochs} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


