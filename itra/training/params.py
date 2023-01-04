import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Model Architechture Configurations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # === text model === #  
    parser.add_argument(
        "--text-model-builder",
        choices=['openclip', 'chineseclip', 'huggingface', 'sbert'], 
        help="use which libarary to build the text backbone",
    )  
    parser.add_argument(
        "--text-model", default='', type=str, 
        help="In open_clip.list_models() or hugging face transformers, depend on text-model-builder",
    )    
    parser.add_argument(
        "--text-model-tag", default='openai', type=str,
        help="Use open_clip.list_pretrained_model_tags(your_model) to check available pretrained weights",
    )     
    parser.add_argument(
        "--text-pooler", type=str, default='cls',
        help="specify pooling streategy for models built by hugging face",
    )
    parser.add_argument(
        "--pretrained-text-model", action="store_true", default=False,
        help="whether load pretrained weigth for the text backbone", 
        # TODO: support build model from scratch for huggingface transformer
    )
    parser.add_argument(
        "--text-head-n-layers", type=int, default=0,
        help="how many MLP layers for text projection head",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=77,
        help="Maximum sequence length for the text backbone",
    )
    # TODO: check the implementation of CoOp-style prompt tuning
    parser.add_argument("--prompt", action="store_true", default=False)
    parser.add_argument("--n-prompt", type=int, default=1)
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--lock-text-model", action="store_true", default=False,
        help="whether include text backbone parameters in the optimizer",
    )
    parser.add_argument(
        "--lock-text-partial", type=str, default='',
        help="list keywords of params names that you want to lock in the text model, seperate by ','. Add '!' at first to do reverse manipulation (partial unfreeze)",
    )
    # === image model === #
    parser.add_argument(
        "--image-model", default='', type=str,
        help="In open_clip.list_models() or torchvision.models",
    )    
    parser.add_argument(
        "--image-model-tag", default='openai', type=str,
    )    
    parser.add_argument(
        "--image-model-builder",
        choices=['openclip', 'chineseclip', 'torchvision', "torchhub"],
        help="use which libarary to build the image backbone",
    ) 
    parser.add_argument(
        "--pretrained-image-model", action="store_true", default=False,
        help="whether load pretrained weigth for the image backbone"
    )
    
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
    ) 
    parser.add_argument(
        "--image-head-n-layers", type=int, default=0,
        help="how many MLP layers for image projection head",
    )
    parser.add_argument(
        "--joint-projection-dim", type=int, default=-1,
        help="dimension of projected representations",
    )
    parser.add_argument(
        "--lock-image-model", action="store_true", default=False,
        help="train image?",
    )
    parser.add_argument(
        "--lock-image-partial", type=str, default='',
        help="list keywords of params names that you want to lock in the image model, seperate by ','. Add '!' at first to do reverse manipulation (partial unfreeze)",
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Knowledge Distillation Configurations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    parser.add_argument("--w-rkd-d", type=float, default=0.5, help="Loss weight.")
    parser.add_argument("--w-rkd-a", type=float, default=1.0, help="Loss weight.")
    parser.add_argument("--dino-teacher-temp", type=float, default=0.04, help="")

    parser.add_argument(
        "--loss",
        type=str,
        default='SimReg',
        help="SimReg, RKD, CompRess, CompRess-1q, InfoNCE, etc.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default='text',
        choices=['text', "image"],
    )
    parser.add_argument(
        "--cache-teacher",
        default=None, 
        type=str,
        )
    parser.add_argument(
        "--cache-dir",
        default='cache/weights', 
        type=str,
        )

    parser.add_argument("--w-simcse", type=float, default=0.0, help="simcse dropout-based contrastive")
    parser.add_argument("--w-distill", type=float, default=1.0, help="")

    
    parser.add_argument("--logit-scale", type=float, default=0.07, help="temperature")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Data and Episodic training
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    parser.add_argument("--nori-dataset", action="store_true", default=False)
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--augmentation",
        choices=[None, "protoclip-light-augmentation"],
        default=None,
        help="Use lighter augmentation for implicit contrast. Choices: [None, protoclip-light-augmentation]",
    ) 
    parser.add_argument(
        "--BYOL",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Path to datasets for evaluation",
    )
    parser.add_argument(
        "--eval-with-wise-ft",
        type=float,
        default=1,
        help="",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Trunck the number of samples in dataset.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    
    parser.add_argument(
        "--images-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        default=0,
        help="Set episode_size to 0 to disable episodic training",
    )  

    parser.add_argument("--retrieval-nori-dataset", action="store_true", default=False)
    parser.add_argument(
        "--retrieval-data",
        type=str,
        default=None,
        help="Path to csv filewith retrieval data",
    )
    parser.add_argument(
        "--retrieval-csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--retrieval-csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--retrieval-csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--retrieval-images-dir",
        type=str,
        default="",
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Projection head
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # parser.add_argument(
    #     "--projection-dim",
    #     type=int,
    #     default=128,
    #     help="dimension of projected representations",
    # ) 
    # parser.add_argument(
    #     "--projection-hidden-dim",
    #     type=int,
    #     default=2048,
    #     help="dimension of projected representations",
    # ) 
    # parser.add_argument(
    #     "--projection-n-layers",
    #     type=int,
    #     default=1,
    #     help="dimension of projected representations",
    # ) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Logging and checkpointing
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Optimization
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # Optimizer parameters
    # From https://github.com/LightDXY/FT-CLIP/blob/da6eab12af754e59c88d490b0ba3835d8f61802b/run_class_finetuning.py#L64
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

        
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
        
    parser.add_argument('--backbone_decay', type=float, default=1)
    parser.add_argument('--layer_decay_image', type=float, default=1)
    parser.add_argument('--layer_decay_text', type=float, default=1)
    # - - - - -

    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 5e-4)')

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--restart", default=False, action="store_true")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    # parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of steps to warmup for.")
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--max-grad-norm",
        default=1e16,
        type=float,
        help="Enable gradient clipping.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Evaluation
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    parser.add_argument(
        "--eval-first",
        default=False,
        action="store_true",
        help="evaluate before start training"
    )
    parser.add_argument("--zeroshot-frequency", type=int, default=0, help="How often to run zero shot.")
    parser.add_argument("--retrieval-frequency", type=int, default=0, help="How often to run coco retrieval.")
    parser.add_argument("--linear-frequency", type=int, default=0, help="How often to run linear eval.")
    parser.add_argument("--nlp-eval-frequency", type=int, default=0, help="How often to run NLP eval.")
    parser.add_argument("--visualize-frequency", type=int, default=-1, help="How often to run linear eval.")
    parser.add_argument("--C", type=float, default=3.16, help="inverse regularizer for logistic reg (sklearn implementation).")
    parser.add_argument(
        "--linear-prob-mode",
        choices=["pytorch", "sklearn"],
        default="pytorch",
        help="Use witch implementation for linear evaluaion"
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Distributed training
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--find-unused-parameters",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args() # to be compatible with ELVATER
    if len(unknown)>0:
        print(f'Unknow args: {unknown}')

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.image_model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
