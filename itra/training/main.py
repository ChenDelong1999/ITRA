
import time
import logging
import os
import random
from datetime import datetime
import numpy as np

import torch
from torch import optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from timm.utils import ModelEma

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from training.model import get_model
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch
from training.optimization import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from evaluation.evaluation import evaluate
from data.train_data import get_data
from loss import get_loss


# to disable warning "huggingface/tokenizers: 
# ("The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...")
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

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            'L' if args.lock_image_model else 'U',
            f'[{args.image_model.replace("/", "_")}-h{args.image_head_n_layers}]',
            'L' if args.lock_text_model else 'U',
            f'[{args.text_model.replace("/", "_")}-h{args.text_head_n_layers}]',
            f"b_{int(args.batch_size * args.world_size)}",
            f"ep_{args.epochs}",
            datetime.now().strftime("%m_%d-%H_%M_%S"),
        ])
    args.name.replace('/', '_')

    args.log_path = None
    if is_master(args, local=args.log_local):
        if os.path.exists(os.path.join(args.logs, args.name)):
            args.name += '-'+datetime.now().strftime("%m_%d-%H_%M_%S")
            print(f"args.name is changed to '{args.name}' to avoid duplication.")
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    
    # NCCL does not support CPU tensor communication. Set up manual multiprocessing communication.
    args.cache_path = os.path.join(args.logs, args.name, "cache") # this is only for protoclip
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path, args.cache_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

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
    # Build teacher, student and loss
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

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        if is_master(args):
            logging.info("Using EMA with decay = %.8f" % args.model_ema_decay)


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
            #broadcast_buffers=False,
            find_unused_parameters=args.find_unused_parameters,
            **ddp_args, 
        )

    loss = get_loss(args)(args, args.joint_projection_dim).to(args.device)
    if is_master(args):
        logging.info(f'Created [{args.loss}] loss: '+str(loss))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # initialize datasets
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.episode_size!=0:
        args.episodic_training=True
        index_mapping = torch.arange(args.episode_size).share_memory_()
        if is_master(args):
            logging.info(f"Model will be trained with episodic training strategy (episodic size={args.episode_size}).")
    else:
        args.episodic_training=False
        index_mapping = None
        if is_master(args):
            logging.info(f"Model will be trained with epoch-wise training strategy.")

    data = get_data(args, (preprocess_train, preprocess_val, preprocess_aug), index_mapping=index_mapping)
    
    if args.train_data is not None:
        if is_master(args):
            logging.info(f'Dataset initialized:')
            logging.info(f'\tdataset n_sample: \t{len(data["train"].dataset)}')
            logging.info(f'\tdataloader n_step: \t{len(data["train"].dataloader)}')
        if args.dataset_size is None:
            args.dataset_size = len(data['train'].dataset)
                
    if not args.episodic_training:
        args.episode_size = args.dataset_size

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # create optimizer and scaler
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    optimizer = None
    scaler = None
    if args.train_data:
        model_without_ddp = model.module if args.distributed else model
        if args.layer_decay_image < 1.0:
            num_layers_image = model_without_ddp.image_backbone.layers
            if is_master(args):
                logging.info(f'Image backbone has {num_layers_image} layers')
            decay = list(args.backbone_decay * args.layer_decay_image ** (num_layers_image + 1 - i) for i in range(num_layers_image + 2))
            decay[-1] /= args.backbone_decay
            assigner_image = LayerDecayValueAssigner(decay)
        else:
            assigner_image = None
            
        if args.layer_decay_text < 1.0:
            num_layers_text = model_without_ddp.text_backbone.layers
            if is_master(args):
                logging.info(f'Text backbone has {num_layers_text} layers')
            decay = list(args.backbone_decay * args.layer_decay_text ** (num_layers_text + 1 - i) for i in range(num_layers_text + 2))
            decay[-1] /= args.backbone_decay
            assigner_text = LayerDecayValueAssigner(decay)
        else:
            assigner_text = None

        # TODO
        # skip_weight_decay_list = model.no_weight_decay()
        skip_weight_decay_list = {'positional_embedding', 'class_embedding', 'logit_scale', 'bn', 'ln', 'bias'}
        
        # TODO
        # args.disable_weight_decay_on_rel_pos_bias = False 
        # if args.disable_weight_decay_on_rel_pos_bias:
        #     for i in range(num_layers):
        #         skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

        optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer_image=assigner_image.get_layer_id if assigner_image is not None else None, 
                get_num_layer_text=assigner_text.get_layer_id if assigner_text is not None else None, 
                get_layer_scale_image=assigner_image.get_scale if assigner_image is not None else None,
                get_layer_scale_text=assigner_text.get_scale if assigner_text is not None else None,
                )


        named_parameters = list(model.named_parameters())
        if is_master(args):
            logging.info(f"Prameters to be optimized:")
            n_trainable_params = 0
            for n, p in named_parameters:
                if p.requires_grad:
                    logging.info(f'\t{n}\t{list(p.size())}')
                    n_trainable_params += p.numel()
            logging.info(f"Prameters NOT to be optimized:")
            n_frozen_params = 0
            for n, p in named_parameters:
                if not p.requires_grad:
                    logging.info(f'\t{n}\t{list(p.size())}')
                    n_frozen_params += p.numel()
                        
        # Original OpenCLIP optimizer implementation
        # exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        # include = lambda n, p: not exclude(n, p)
        # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        # optimizer = optim.AdamW(
        #     [
        #         {"params": gain_or_bias_params, "weight_decay": 0.},
        #         {"params": rest_params, "weight_decay": args.wd},
        #     ],
        #     lr=args.lr,
        #     betas=(args.beta1, args.beta2),
        #     eps=args.eps,
        # )

        scaler = GradScaler() if args.precision == "amp" else None
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    if is_master(args):
        model_without_ddp = model.module if args.distributed else model
        logging.info('Model\n' +str(model))
        logging.info(f'Total Model Parameters (M):\t        {round(sum(p.numel() for p in model_without_ddp.parameters())/1e6, 2)}')
        logging.info(f'Image Backbone Parameters (M):\t     {round(sum(p.numel() for p in model_without_ddp.image_backbone.parameters())/1e6, 2)}')
        logging.info(f'Text Backbone Parameters (M):\t      {round(sum(p.numel() for p in model_without_ddp.text_backbone.parameters())/1e6, 2)}')
        logging.info(f'Image Projection Parameters (M):\t   {round(sum(p.numel() for p in model_without_ddp.image_projection_head.parameters())/1e6, 2)}')
        logging.info(f'Text Projection Parameters (M):\t    {round(sum(p.numel() for p in model_without_ddp.text_projection_head.parameters())/1e6, 2)}')
        if args.train_data:
            logging.info(f'Trainable Parameters (M):\t{round(n_trainable_params/1e6, 2)}')
            logging.info(f'Frozen Parameters (M):\t{round(n_frozen_params/1e6, 2)}')


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # optionally resume from a checkpoint
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint from '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
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
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="vision-language-knowledge-distillation",
            notes=args.name,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        #wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, start_epoch, preprocess_val, args, writer)
        return
    
    if args.eval_first:
        evaluate(model_ema.ema if args.model_ema else model, start_epoch, preprocess_val, args, writer)
    
    if is_master(args):
        logging.info("args:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Start training loop
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    profiling = {
        "epsidoe model training time (m)": 0,
    }
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        
        if args.episodic_training:
            # Random episode sampling      
            index_mapping[:] = torch.from_numpy(np.random.choice(args.dataset_size, args.episode_size, replace=True))
            if is_master(args):
                logging.info(f"Randomly select {args.episode_size} samples from full dataset {args.dataset_size} as current episode.")
        
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
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': epoch})

        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
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
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb', 'cache', 'features', 'weights', 'vision_benchmark'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
