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

from utils.training_utils import AverageMeter, Cacher
from .distributed import is_master, gather_features, get_gathered_item
from loss import NEED_LOGIT_SCALE, NEED_GATHER, NEED_PROTOTYPE_LAYER

from loss import CLIPLoss

def train_one_epoch(
    model, 
    model_ema, 
    data, 
    epoch, 
    optimizer, 
    scaler, 
    scheduler,  
    loss, 
    args, 
    writer 
    ):
    device = torch.device(args.device) 
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    forward_context = torch.no_grad if args.cache_teacher is not None else suppress

    model.train()  
    model_without_ddp = model.module if args.distributed else model
    
    if args.lock_text_model and args.adapter is None:
        model_without_ddp.text_backbone.eval()
        if is_master(args):
            logging.info('set text backbone to .eval() mode')
    if args.lock_image_model:
        model_without_ddp.image_backbone.eval()
        if is_master(args):
            logging.info('set image backbone to .eval() mode')

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    if args.cache_teacher is not None and is_master(args):
        cacher = Cacher(n_sample=args.episode_size, n_dim=model_without_ddp.text_dim, cache_file=args.cache_teacher)
        logging.info(f'Preparing to cache teacher features, size={cacher.feature.size()}, to be saved to {args.cache_teacher}')

    if args.w_simcse > 0:
        simcse_contrastive_loss = CLIPLoss(args, dim=0)

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        index, images, texts, labels = batch
        all_labels = get_gathered_item(labels.cuda(), args)
        all_index = get_gathered_item(index.cuda(), args)
        
        if len(index)!=args.batch_size and args.cache_teacher is None:
            continue  # drop the last incomplete batch if train        

        if args.BYOL:
            images, images_aug = images
            images_aug = images_aug.to(device=device, non_blocking=True)
            MSE = nn.MSELoss()
        if args.cache_teacher is None:
            images = images.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)

        # # # # # # # # # # # # # # # # # # 
        # model forward
        # # # # # # # # # # # # # # # # # # 
        with autocast():
            with forward_context():
                if args.loss in NEED_PROTOTYPE_LAYER:
                    if args.teacher=='text':
                        w = model_without_ddp.image_projection_head.last_layer.weight_v.data.clone()
                        model_without_ddp.text_projection_head.last_layer.weight_v.data.copy_(w)
                    elif args.teacher=='image':
                        w = model_without_ddp.text_projection_head.last_layer.weight_v.data.clone()
                        model_without_ddp.image_projection_head.last_layer.weight_v.data.copy_(w)

                image_features, text_features, logit_scale = model(images, texts, text_only=(args.cache_teacher is not None) or args.w_distill==0)

                if args.w_simcse > 0:
                    text_features_2 = model_without_ddp.encode_text(texts, projection=True)
                    
                # gather features
                if args.distributed and args.loss in NEED_GATHER:
                    all_image_features, all_text_features = gather_features(
                        image_features, text_features,
                        args.local_loss, args.gather_with_grad, 
                        args.rank, args.world_size, args.horovod
                        )
                    if args.w_simcse > 0:
                        all_text_features_2 = get_gathered_item(text_features_2, args)
                else:
                    all_image_features = image_features
                    all_text_features = text_features
                    if args.w_simcse > 0:
                        all_text_features_2 = text_features_2

                if args.BYOL:
                    image_features_aug = model_without_ddp.encode_image(images_aug, projection=True)
                    ssl_loss = MSE(image_features_aug, image_features.detach())
                elif args.w_simcse > 0:
                    ssl_loss = args.w_simcse * simcse_contrastive_loss(all_text_features, all_text_features_2, logit_scale=np.exp(2.996))
                else:
                    ssl_loss = 0
                
                if args.teacher=='text':
                    teacher_features, student_features = all_text_features, all_image_features
                elif args.teacher=='image':
                    teacher_features, student_features = all_image_features, all_text_features
                
                if args.loss in NEED_LOGIT_SCALE:
                    if args.loss in ['UniCL', 'CrossEntropy']:
                        align_loss = loss(teacher_features, student_features, logit_scale=logit_scale, labels=all_labels)
                    else:
                        align_loss = loss(teacher_features, student_features, logit_scale=logit_scale)
                else:
                    align_loss = loss(teacher_features, student_features)            
                total_loss = args.w_distill * align_loss + ssl_loss
        
        if args.cache_teacher is not None:
            # # # # # # # # # # # # # # # # # # 
            # cache teacher
            # # # # # # # # # # # # # # # # # # 
            if is_master(args):
                cacher.load_batch(all_index, all_text_features)

                batch_size = len(images)
                batch_count = i + 1
                num_samples = batch_count * batch_size * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch
                sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
                logging.info(
                        f"Caching Features [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {total_loss.item():.5f} "
                        f"index: {all_index.size()} "
                        f"features: {all_text_features.size()} "
                    )
        else:                
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
                    "loss-distill": align_loss.item(),
                    "loss-ssl": ssl_loss.item() if args.BYOL or args.w_simcse > 0 else 0,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient-norm": norm,
                    "logit_scale": model.module.logit_scale.item() if args.distributed else model.logit_scale.item()
                }
                profiling = {
                    "batch data time (s)": data_time_m.val,
                    "bathc total time (s)": batch_time_m.val,
                }
            
                if args.prompt:
                    log_data['prompt-norm'] = model.module.prompt.norm(dim=1, p=2).mean() if args.distributed else model.prompt.norm(dim=1, p=2).mean() 
                    
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

    
    if args.cache_teacher is not None:
        if is_master(args):
            logging.info(f'Saving features...\n{str(cacher.feature)}')
            cacher.save()
            logging.info(f'Saved.')
        if args.distributed:
            torch.distributed.barrier()
        exit(1)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

