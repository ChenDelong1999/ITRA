import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from training.params import parse_args
import argparse
#from training.evaluations.evaluation import evaluate
from training.projection import add_projection_head
#from open_clip import create_model_and_transforms
import logging
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
#from openTSNE import TSNE
from training.visual_model import get_visual_model_and_preprocess

import torch.nn.functional as F
# to disable warning "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from refile import smart_open
    import nori2 as nori
    import io
except ImportError:
    # TODO: remove nori dependency when publish codes
    pass

import matplotlib.pyplot as plt
from openTSNE import TSNE


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key='filepath', caption_key='title', sep="\t", dataset_size=None, index_mapping=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        if input_filename[:2]=='s3':
            self.using_nori = True
            df = pd.read_csv(smart_open(input_filename, "r"), sep=sep)
            self.f = None
        else:
            self.using_nori = False
            df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        self.transforms = transforms
        self.inversed_normalize = Compose([
            Normalize((0.0, 0.0, 0.0), (1/0.26862954, 1/0.26130258, 1/0.27577711)),
            Normalize((-0.48145466, -0.4578275, -0.40821073), (1.0, 1.0, 1.0)),
            ])

        # Faster data loading. see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.images = np.array(df[img_key].tolist()).astype(np.string_)
        self.captions = np.array(df[caption_key].tolist())
        for i in range(len(self.captions)):
            self.captions[i] = self.captions[i].encode('ascii',errors='ignore')
        self.captions = self.captions.astype(np.string_)

        # use a subset of given dataset
        if dataset_size is not None:
            self.images = self.images[:dataset_size]
            self.captions = self.captions[:dataset_size]
        
        if index_mapping is None:
            self.index_mapping=torch.arange(len(self.captions))
        else:
            self.index_mapping = index_mapping
                
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, episodic_index):
        index = self.index_mapping[episodic_index]

        #images = self.transforms(Image.open(str(self.images[index])))
        if self.using_nori:
            if self.f is None:
                self.f = nori.Fetcher()
            image = Image.open(io.BytesIO(self.f.get(self.images[index].decode('utf-8'))))
        else:
            image = Image.open(str(self.images[index].decode('utf-8')))
        
        image = self.transforms(image)
        texts = str(self.captions[index].decode('utf-8'))
        
        return episodic_index, image, texts[:100]# FIXME: '[:100]' is a temperate solution of CLIP's tokenizer overlength bug
    
    def get_data(self, episode_index):
        idx = self.index_mapping[episode_index]
            
        # get image data
        if self.using_nori:
            if self.f is None:
                self.f = nori.Fetcher()
            pic = Image.open(io.BytesIO(self.f.get(self.images[idx].decode('utf-8'))))
        else:
            pic = Image.open(str(self.images[idx].decode('utf-8')))
        
        
        image = self.inversed_normalize(self.transforms(pic))
        texts = self.captions[idx].decode('utf-8')
        
        return image, texts



def show_tsne(image_features, text_features, file_name, title):

    logging.info('Fitting T-SNE')
                    
    tsne_img = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(image_features)
    tsne_text = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(text_features)
    tsne_all = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(torch.cat([image_features, text_features], dim=0))
    
    plt.figure(figsize=(75,25))
    plt.rc('font', size=25) 
    plt.subplots_adjust(top=0.9,wspace=0.05,hspace=0.05)

    plt.subplot(131)
    plt.xticks([])
    plt.yticks([])
    plt.title('image features')
    plt.scatter(tsne_img[:,0], tsne_img[:,1], s=1.5, c='red', alpha=0.8)

    plt.subplot(132)
    plt.xticks([])
    plt.yticks([])
    plt.title('image-text features')
    plt.scatter(tsne_all[:len(image_features),0], tsne_all[:len(image_features),1], s=1, c='red', alpha=0.5)
    plt.scatter(tsne_all[len(image_features):,0], tsne_all[len(image_features):,1], s=1, c='blue', alpha=0.5)

    plt.subplot(133)
    plt.xticks([])
    plt.yticks([])
    plt.title('text features')
    plt.scatter(tsne_text[:,0], tsne_text[:,1], s=1.5, c='blue', alpha=0.8)
    
    plt.suptitle(title)
    plt.savefig(file_name, bbox_inches='tight')

    logging.info(f'T-SNE visuallization saved to: {file_name}')


def extract_feature(student, teacher, dataset, args):
    dataloader = DataLoader(dataset, batch_size=100, num_workers=8, persistent_workers=True)
    all_image_features = []
    all_text_features = []
    for (index, images, texts) in tqdm(dataloader):
        with torch.no_grad():
            raw_text_features = teacher.encode(
                    texts,
                    convert_to_tensor=True, 
                    show_progress_bar=False
                    ).detach()
            if args.add_teacher_projection_head:
                text_features = student.text_projection_head(raw_text_features) + raw_text_features if args.res_teacher_projection else student.text_projection_head(raw_text_features)
            else:
                text_features = raw_text_features

            raw_image_features = student(images.cuda())    
            image_features = student.image_projection_head(raw_image_features)      
        
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        all_image_features.append(image_features)
        all_text_features.append(text_features)
    
    all_image_features = torch.stack(all_image_features).view(-1, args.projection_dim)
    all_text_features = torch.stack(all_text_features).view(-1, args.projection_dim)

    # print(all_image_features.size(), all_text_features.size())
    # np.save(arr=[all_image_features.cpu().numpy(), all_text_features.cpu().numpy()], file='cache/features.npy')
    # exit()
    return all_image_features.cpu(), all_text_features.cpu()


def evaluate_checkpoint(checkpoint_path, epoch, args):
    # load model
    
    logging.info(f'Loading pretrained text trasformer teacher: {args.pretrained_text}.')
     
    student, preprocess_train, preprocess_val = get_visual_model_and_preprocess(args)   
    student = add_projection_head(student, student.output_dim, args)

    teacher = SentenceTransformer(args.pretrained_text)
    teacher.to(device=args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    student.load_state_dict(sd)
    logging.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")  
    
    student = student.to(device)
    teacher = teacher.to(device)

    dataset = CsvDataset(args.input_filename, preprocess_val, dataset_size=args.num_points)
    image_features, text_features = extract_feature(student, teacher, dataset, args)
    show_tsne(image_features, text_features, file_name=os.path.join(args.exp_dir, 'visualization', f'tsne({len(dataset)})_epoch_{epoch}.png'), title=args.exp_dir)
    

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

    args.input_filename = 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv'
    args.num_points = 200000
    args.exp_dir = exp_dir

    args.zeroshot_frequency = 1
    args.linear_frequency = 1
    args.retrieval_frequency = 1
    args.save_logs = False
    args.distributed = False
    args.wandb = False
    args.rank = 0
    args.batch_size = 32
    args.workers = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loaded params from file '{params_file}':")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")
    
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

                    evaluate_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, args=args)
                    
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
        
        evaluate_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, args=args)
        
