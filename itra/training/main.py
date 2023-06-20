import time
import logging
import os
import random
import numpy as np
import yaml
import wandb
import torch.utils.tensorboard as tensorboard

import argparse

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from timm.utils import ModelEma

from model.models import get_model
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging, get_exp_name
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch
from training.optimization import get_optimizer

from evaluation.evaluation import evaluate
from data.train_data import get_data
from data.episodic_training import init_index_mapping, update_index_mapping
from loss import get_loss

# to disable warning "huggingface/tokenizers: 
#   ("The current process just got forked, after parallelism has already been used. 
#   Disabling parallelism to avoid deadlocks...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# fix random seed
def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)



def main():
    args = parse_args()
    random_seed(args.seed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Configurate distributed training and logging
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.name = get_exp_name(args)
    args.log_path = None

    # Set logger
    if is_master(args, local=args.log_local):        
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    # init wandb & tensorboard logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")


    # NCCL does not support CPU tensor communication (or large GPU tensor that cause cuda OOM). 
    # Set up manual multiprocessing communication (for ProtoCLP).
    args.cache_path = os.path.join(args.logs, args.name, "cache") 
    if is_master(args):
        for dirname in [args.tensorboard_path, args.checkpoint_path, args.cache_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        # you will have to configure this for your project!
        wandb.init(
            project="ITRA",
            notes=args.name,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')

    if args.copy_codebase and is_master(args):
        copy_codebase(args)

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Build model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # Set barriers to avoid multiple downloads of pretrained weights
    if args.pretrained_text_model or args.pretrained_image_model and args.distributed:
        if is_master(args):
            model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)
        if args.distributed:
            dist.barrier()   
        if not is_master(args): 
            model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)
        if args.distributed:
            dist.barrier()   
    else:
        model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)

    # Model Exponential Moving Average (only for evaluation now)
    model_ema = None
    if is_master(args):
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '')
            logging.info("Using EMA with decay = %.8f" % args.model_ema_decay)

    # Convert to sync BN and model DDP
    print("args.distributed", args.distributed)
    print("not args.horovod", not args.horovod)
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device], 
            # broadcast_buffers=False,
            find_unused_parameters=args.find_unused_parameters,
            **ddp_args, 
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Get loss function
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    loss = get_loss(args)(args, args.joint_projection_dim).to(args.device)
    logging.info(f'Using [{args.loss}] loss: ' + str(loss))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # initialize datasets
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    index_mapping = init_index_mapping(args)
    data = get_data(args, (preprocess_train, preprocess_val, preprocess_aug), index_mapping)
    
    if args.train_data is not None and args.dataset_size is None:
        args.dataset_size = len(data['train'].dataset)
                
    if not args.episodic_training:
        args.episode_size = args.dataset_size

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # create optimizer, scaler, and scheduler
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    optimizer = None
    scaler = None
    scheduler = None
    if args.train_data is not None:
        optimizer = get_optimizer(model, args)
        scaler = GradScaler() if args.precision == "amp" else None
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    if is_master(args):
        model_without_ddp = model.module if args.distributed else model
        named_parameters = list(model.named_parameters())
        logging.info(f"↓ Prameters to be optimized ↓")
        n_trainable_params = 0
        for n, p in named_parameters:
            if p.requires_grad:
                logging.info(f'\t{n}\t{list(p.size())}')
                n_trainable_params += p.numel()
        logging.info(f"---")
                
        logging.info(f"↓ Prameters NOT to be optimized ↓")
        n_frozen_params = 0
        for n, p in named_parameters:
            if not p.requires_grad:
                logging.info(f'\t{n}\t{list(p.size())}')
                n_frozen_params += p.numel()
        logging.info(f"---")

        logging.info('Model structure\n' +str(model))
        logging.info(f'Total Model Parameters (M):\t        {round(sum(p.numel() for p in model_without_ddp.parameters())/1e6, 2)}')
        logging.info(f'Image Backbone Parameters (M):\t     {round(sum(p.numel() for p in model_without_ddp.image_backbone.parameters())/1e6, 2)}')
        logging.info(f'Text Backbone Parameters (M):\t      {round(sum(p.numel() for p in model_without_ddp.text_backbone.parameters())/1e6, 2)}')
        logging.info(f'Image Projection Parameters (M):\t   {round(sum(p.numel() for p in model_without_ddp.image_projection_head.parameters())/1e6, 2)}')
        logging.info(f'Text Projection Parameters (M):\t    {round(sum(p.numel() for p in model_without_ddp.text_projection_head.parameters())/1e6, 2)}')
        logging.info(f'Trainable Parameters (M):\t{round(n_trainable_params/1e6, 2)}')
        logging.info(f'Frozen Parameters (M):\t{round(n_frozen_params/1e6, 2)}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # optionally resume from a checkpoint
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint from '{args.resume}'...")
            # checkpoint = torch.load(args.resume, map_location=device)
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                msg = model.load_state_dict(sd, strict=False)
                if is_master(args):
                    logging.info(msg)
                try:
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                except ValueError:
                    logging.info('optimizer param groups do not mathch. Skip resuming optimizer.')
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.restart:
        start_epoch = 0
        # if args.loss in NEED_LOGIT_SCALE:
        #     model.module.reinit_logit_scale(args.logit_scale) if args.distributed else model.reinit_logit_scale(args.logit_scale)
        #     logging.info(f'logict scale re-initialized to {args.logit_scale}')
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Evaluation only or evaluation before training
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if 'train' not in data:
        evaluate(model, start_epoch, preprocess_val, args, writer)
        return
    
    if args.eval_first:
        evaluate(model_ema.ema if args.model_ema else model, start_epoch, preprocess_val, args, writer)
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Save arguments and configurations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if is_master(args):
        logging.info("args:")
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")

        with open(os.path.join(args.logs, args.name, "params.yml"), 'w') as f:
            yaml.dump(vars(args), f)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Start training loop
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    profiling = {"epsidoe model training time (m)": 0}
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        
        # Random episode sampling      
        if args.episodic_training:
            index_mapping = update_index_mapping(index_mapping, args)

        start = time.time()
        train_one_epoch(
            model=model,
            model_ema=model_ema,
            data=data,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            loss=loss,
            args=args,
            writer=writer
            )

        if is_master(args):
            duration = (time.time()-start)/60
            profiling['epsidoe model training time (m)'] = duration
            logging.info(f'[Profiling] Model training finished in {duration:.2f} minute.')
            
            for name, val in profiling.items():
                name = "profiling/" + name
                if writer is not None:
                    writer.add_scalar(name, val, epoch)
                if args.wandb:
                    wandb.log({name: val, 'step': epoch})

        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
            if is_master(args):
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.model_ema else model_ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )
                if args.save_most_recent:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                    )            
            evaluate(model if not args.model_ema else model_ema.ema, completed_epoch, preprocess_val, args, writer)

        if args.distributed:
            dist.barrier()

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    args.name.replace('/', '_')
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)

    # TODO validate gitingnore_patterns implementation
    gitingnore_patterns = open('.gitignore').read().split('\n')
    print(f'load ignore patterns from gitignore: {gitingnore_patterns}')
    copytree(current_code_path, new_code_path, ignore=ignore_patterns(*gitingnore_patterns)) 

    # copytree(current_code_path, new_code_path, ignore=ignore_patterns('logs', 'wandb', 'cache', 'features', 'weights', 'vision_benchmark'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
