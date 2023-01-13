import os
import torch
from training.params import parse_args
import argparse
import logging
from model.model import get_model
# logger = logging.getLogger('mylogger') 

# to disable warning "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_params(params_file, args):
    args = vars(args)
    with open(params_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(': ')
            key, value = line[0], ''.join(line[1:])
            if key in args.keys() and args[key] is not None:
                #print(key, value, args[key], type(args[key]))
                args[key] = type(args[key])(value)
            else:
                args[key] = value
            if value == 'False':
                args[key] = False
            if value == 'None':
                args[key] = None
    return argparse.Namespace(**args)


def get_model_from_log_dir(exp_dir, epoch='latest'):
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    params_file = os.path.join(exp_dir, 'params.txt')
    
    args = parse_args()
    args = load_params(params_file, args)

    args.nlp_eval_frequency = 1
    args.zeroshot_frequency = 0
    args.linear_frequency = 0
    #args.linear_prob_mode= 'pytorch-search'
    args.retrieval_frequency = 0
    args.save_logs = False
    args.distributed = False
    args.wandb = False
    args.rank = 0
    args.batch_size = 32
    args.workers = 12
    args.image_teacher='none'
    args.pretrained_text_model = False
    args.pretrained_image_model = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # logger.info(f"Loaded params from file '{params_file}':")
    # for name in sorted(vars(args)):
    #     val = getattr(args, name)
    #     logger.info(f"  {name}: {val}")
    

    # logger.info(f'evaluate single checkpoint: {checkpoint}')
    checkpoint = f'epoch_{epoch}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    
    model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    # logger.info(str(msg))
    # logger.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")  

    return model

