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
from training.prompt import encode_text_with_prompt

def train_one_epoch(
    model, 
    data, 
    epoch, 
    optimizer, 
    scaler, 
    scheduler,  
    distiller, 
    args, 
    writer 
    ):
    device = torch.device(args.device) 
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()  
    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
        
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        index, images, texts = batch
        if len(index)!=args.batch_size:
            continue  # drop the last incomplete batch
        
        #all_index = get_gathered_item(index.cuda(), args)
        images = images.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # # # # # # # # # # # # # # # # # # 
        # model forward
        # # # # # # # # # # # # # # # # # # 
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            # gather features
            if args.distributed:
                all_image_features, all_text_features = gather_features(
                    image_features, text_features,
                    args.local_loss, args.gather_with_grad, 
                    args.rank, args.world_size, args.horovod
                    )
            total_loss = distiller(all_text_features, all_image_features, logit_scale=logit_scale)
            
        # # # # # # # # # # # # # # # # # # 
        # loss backward
        # # # # # # # # # # # # # # # # # # 
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
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
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

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
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient-norm": norm,
                "logit_scale": model.module.logit_scale.item()
            }
            profiling = {
                "batch data time (s)": data_time_m.val,
                "bathc total time (s)": batch_time_m.val,
            }
        
            if args.prompt:
                log_data['prompt-norm'] = model.prompt().norm(dim=1, p=2).mean()

                # if args.unlock_text_teacher:
                #     log_data['text-teacher-grad-norm'] = text_teacher_norm
                
            for name, val in log_data.items():
                name = "training/" + name
                if writer is not None:
                    writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            for name, val in profiling.items():
                name = "profiling/" + name
                if writer is not None:
                    writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

