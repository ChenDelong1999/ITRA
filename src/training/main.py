
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

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import trace_model, create_model_and_transforms, create_transforms, list_models
from sentence_transformers import SentenceTransformer
from training.visual_model import get_visual_model_and_preprocess

from seed import models
# from training.clustering import Clustering
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, cosine_wd
from training.train import train_one_epoch
from training.evaluations.evaluation import evaluate
from training.projection import add_projection_head, get_adaption_head

from utils import protoseed_utils
from distiller import get_distiller

import matplotlib.pyplot as plt

# to disable warning "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    args = parse_args()
    random_seed(args.seed)

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M"), # disabled since it might make different process to have different names
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
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
    args.cache_path = os.path.join(args.logs, args.name, "cache") 
    args.visualization_path = os.path.join(args.logs, args.name, "visualization") 
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path, args.cache_path, args.visualization_path]:
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
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Build teacher and student
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # === text teacher === #
    logging.info(f'Loading pretrained text transformer as teacher: {args.text_teacher}.')
    if args.text_teacher!='none':
        text_teacher = SentenceTransformer(args.text_teacher)
        if args.text_teacher=='clip-ViT-B-32':
            args.text_teacher_dim = 512
        else:   
            args.text_teacher_dim = text_teacher.get_sentence_embedding_dimension()
        distiller_text = get_distiller(args.distiller)(args, args.text_teacher_dim).to(args.device)
    else:
        text_teacher = None
        distiller_text = None
        args.text_teacher_dim = 1
    
    if args.adaption_head:
        adaption_head = get_adaption_head(args)
    else:
        adaption_head = None


    # === image teacher === #
    #if args.image_teacher in list_models():
    if args.image_teacher!='none':
        CLIP_model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.image_teacher,
            pretrained='openai',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            args=args
        )
        image_teacher = CLIP_model.visual
        args.image_teacher_dim = 1024 # FIXME
        distiller_image = get_distiller(args.distiller)(args, args.image_teacher_dim).to(args.device)
    else:
        image_teacher = None
        args.image_teacher_dim = 1
        distiller_image = None

    # === student === #
    if args.model in list_models():
        CLIP_model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            args=args
        )
        # CLIP_model created by OpenCLIP has image and text tower,
        # remove text tower and leave the image tower as student.
        student = CLIP_model.visual
    else:
        pretrained = (args.pretrained=='torchvision')
        logging.info(f'[torchvision]: loading {args.model} model as student, pretrained={pretrained}')
        student = models.__dict__[args.model](pretrained=pretrained, num_classes=1000)
        student.output_dim = student.fc.weight.shape[1]
        student.fc=torch.nn.Identity()
        student.to(device=args.device)
        
    preprocess_train, preprocess_val = create_transforms(image_size=224, args=args)

    if args.freeze_student_backbone:
        for param in student.parameters():
            param.requires_grad = False
            
    student = add_projection_head(student, student.output_dim, args)

    if is_master(args):
        logging.info('* '*32 + '[student]\n' +str(student))
        logging.info('* '*32 + '[image_teacher]\n' +str(image_teacher))
        logging.info('* '*32 + '[text_teacher]\n' +str(text_teacher))
        logging.info('* '*32 + '[adaption_head]\n' +str(adaption_head))

    if args.trace:
        student = trace_model(student, batch_size=args.batch_size, device=device)

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        student = torch.nn.parallel.DistributedDataParallel(
            student, 
            device_ids=[device], 
            broadcast_buffers=False,
            **ddp_args, 
        )

    
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

    data = get_data(args, (preprocess_train, preprocess_val), index_mapping=index_mapping)

    if args.train_data is not None and args.dataset_size is None:
        args.dataset_size = len(data['train'].dataset.captions)
    if not args.episodic_training:
        args.episode_size = args.dataset_size

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # create optimizer and scaler
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(student.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=student.named_parameters())
            hvd.broadcast_parameters(student.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

        scheduler = None
        if 'train' in data and optimizer is not None:
            total_steps = data["train"].dataloader.num_batches * args.epochs
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

        # if use gradually disappering adaptive teacher projection head
        if args.adaption_head:
          
            adaption_head_optimizer = optim.AdamW(
                [{"params": adaption_head.parameters(), "weight_decay": args.wd}],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            panalty_scheduler = cosine_wd(base_value=args.base_panalty_weight, final_value=args.final_panalty_weight, power=args.quiting_power, total_step=args.epochs * data["train"].dataloader.num_batches)
            if is_master(args):
                fig_path = os.path.join(args.visualization_path, 'panalty_scheduler.png')
                plt.plot(panalty_scheduler)
                plt.savefig(fig_path)
                logging.info(f'panalty_scheduler (graph saved to {fig_path}):\n'+str(panalty_scheduler))
        else:
            adaption_head_optimizer = None
            panalty_scheduler=None

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # optionally resume from a checkpoint
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                student.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                student.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

        
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
            project="TrainProtoRKD",
            notes=args.name,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(student, log='all')
        #wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(student, text_teacher, start_epoch, preprocess_val, args, writer)
        return
    
    if is_master(args):
        logging.info("Params:")
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
            student, text_teacher, image_teacher, 
            data, epoch, optimizer, scaler, scheduler, 
            distiller_text, distiller_image, args, writer, 
            adaption_head=adaption_head, adaption_head_optimizer=adaption_head_optimizer, panalty_scheduler=panalty_scheduler
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
                "state_dict": student.state_dict(),
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

        evaluate(student, text_teacher, completed_epoch, preprocess_val, args, writer)
        if args.distributed:
            dist.barrier()

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
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
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb', 'cache', 'features', 'weights'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
