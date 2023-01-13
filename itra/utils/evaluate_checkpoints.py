import os
import time
import pandas as pd
import torch
from training.params import parse_args
import argparse
from evaluation.evaluation import evaluate
import logging
import matplotlib.pyplot as plt
#from openTSNE import TSNE
from model.model import get_model

# to disable warning "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

def evaluate_checkpoint(checkpoint_path, epoch, args):
    # load model

    model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    logging.info(str(msg))
    logging.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")  
    
    metrics = evaluate(model, epoch, preprocess_val, args, tb_writer=None, fast_evaluation=False)    
    return metrics

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



if __name__ == '__main__':
    exp_dir = input('Please input your experiment dir: ')
    single_eval = input('Specify a checkpoint epoch? (press "enter" to scan and evaluate all checkpoints) ')
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    params_file = os.path.join(exp_dir, 'params.txt')
    
    args = parse_args()
    args = load_params(params_file, args)

    args.nlp_eval_frequency = 1
    args.zeroshot_frequency = 1
    args.linear_frequency = 1
    #args.linear_prob_mode= 'pytorch-search'
    args.retrieval_frequency = 1
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

    logging.info(f"Loaded params from file '{params_file}':")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")
    
    all_metrics = pd.DataFrame()

    if not single_eval:
        finished = ['epoch_latest.pt']
        while True:
            checkpoints = os.listdir(checkpoint_dir)
            for checkpoint in checkpoints:
                if checkpoint not in finished:
                    logging.info(f'found new checkpoint: {checkpoint}')
                    time.sleep(10) # in case of the checkpoint is not fully written to disk
                    epoch = int(checkpoint.split('_')[1][:-3])
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

                    metrics = evaluate_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, args=args)
                    metrics['epoch'] = epoch

                    for key in metrics.keys():
                        metrics[key] = [metrics[key]]
                    metrics = pd.DataFrame.from_dict(metrics)

                    all_metrics = pd.concat([all_metrics, metrics])
                    all_metrics.to_csv(os.path.join(exp_dir, 'evaluation_metrics_all.csv'))
                    # all_metrics.to_csv(os.path.join(exp_dir, f'evaluation_metrics@epoch_{epoch}.csv'))
                    print(all_metrics)

                    finished.append(checkpoint)
            time.sleep(10)
    else:
        checkpoint = f'epoch_{single_eval}.pt'
        logging.info(f'evaluate single checkpoint: {checkpoint}')
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        if single_eval=='latest':
            epoch = -1
        else:
            epoch = int(checkpoint.split('_')[1][:-3])
        
        metrics = evaluate_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, args=args)
        metrics['epoch'] = epoch
        for key in metrics.keys():
            metrics[key] = [metrics[key]]
        metrics = pd.DataFrame.from_dict(metrics)

        all_metrics = pd.concat([all_metrics, metrics])
        all_metrics.to_csv(os.path.join(exp_dir, f'single_evaluation_metrics@epoch_{epoch}.csv'))
        print(all_metrics)

