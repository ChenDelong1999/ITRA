import logging
import torch
import numpy as np

def init_index_mapping(args):

    if args.episode_size!=0:
        args.episodic_training=True
        index_mapping = torch.arange(args.episode_size).share_memory_()
        logging.info(f"Model will be trained with episodic training strategy (episodic size={args.episode_size}).")
    else:
        args.episodic_training=False
        index_mapping = None
        logging.info(f"Model will be trained with epoch-wise training strategy.")
        
    return index_mapping


def update_index_mapping(index_mapping, args):
    # Random episode sampling      
    index_mapping[:] = torch.from_numpy(np.random.choice(args.dataset_size, args.episode_size, replace=True))
    logging.info(f"Randomly select {args.episode_size} samples from full dataset {args.dataset_size} as current episode.")
    return index_mapping