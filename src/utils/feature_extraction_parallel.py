import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import logging
import math
import numpy as np

from training.data import get_data
from training.params import parse_args
from training.logger import setup_logging
from training.distributed import is_master, init_distributed_device, world_info_from_env, gather_features, get_gathered_item



def main():
    args = parse_args()
    
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(False, args.log_level)

    logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
            )
    
    model = torchvision.models.resnet50(pretrained=True)
    model.to(args.device)
    model.eval()
    model.fc = nn.Identity()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def _convert_to_rgb(image):
        return image.convert('RGB')
    preprocess = transforms.Compose([
        _convert_to_rgb,
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    if is_master(args):
        logging.info(model)
        logging.info(preprocess)
    
    data = get_data(args, (preprocess, preprocess, preprocess), index_mapping=None)
    if is_master(args):
        logging.info(f'Dataset initialized:')
        logging.info(f'\tdataset length: \t{len(data["train"].dataset)}')
        logging.info(f'\tdataloader length: \t{len(data["train"].dataloader)}')
        logging.info(f'\tsampler length: \t{len(data["train"].sampler)}')
        dataset_features = np.zeros([len(data["train"].dataset), 2048])

    for i, batch in enumerate(data['train'].dataloader):
        index, images, texts = batch
        images = images.to(device=device, non_blocking=True)
        all_index = get_gathered_item(index.cuda(), args)
        with torch.no_grad():
            features = model(images)
            all_features = get_gathered_item(features, args)
        
        if is_master(args):
            dataset_features[all_index.cpu().numpy()] = all_features.cpu().numpy()
            if (i % 10 == 0):
                batch_size = len(images)
                num_samples = i * batch_size * args.world_size
                samples_per_epoch = data['train'].dataloader.num_samples
                percent_complete = 100.0 * i / data['train'].dataloader.num_batches
                sample_digits = math.ceil(math.log(data['train'].dataloader.num_samples + 1, 10))

                logging.info(
                    f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                )
                
    if is_master(args):
        logging.info(dataset_features)
        logging.info(f'Saving features {dataset_features.shape} to: "{args.cache_teacher}".')
        np.save(args.cache_teacher, dataset_features)
        logging.info('saved.')
    
    
    torch.distributed.barrier()


'''
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m utils.feature_extraction_parallel \
    --train-data 'cache/yfcc_nori.csv' --batch-size 1024 --cache-teacher 'cache/resnet50-yfcc14m.npy'

'''

if __name__ == "__main__":
    main()
