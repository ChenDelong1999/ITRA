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
from seed import models
from open_clip import trace_model, create_model_and_transforms, create_transforms, list_models

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
import random
try:
    from refile import smart_open
    import nori2 as nori
    import io
except ImportError:
    # TODO: remove nori dependency when publish codes
    pass

import matplotlib.pyplot as plt
from openTSNE import TSNE
from training.evaluations.zero_shot import imagenet_classnames, imagenet_templates

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



class ImageNet_nori(Dataset):
    
    def __init__(self, transform, split='val'):

        super(ImageNet_nori, self).__init__()
        if split=='train':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.train.nori.list"
        elif split=='val':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.val.nori.list"

        self.f = None #nori.Fetcher()
        self.f_list = []
        self.transform = transform

        with smart_open(nori_path, "r") as f:
            for line in f:
                self.f_list.append(line.strip().split())

        self.labels = []
        for i in self.f_list:
            self.labels.append(int(i[1]))
        self.labels = self.labels

              
    def __getitem__(self, idx):
        if self.f is None:
            self.f = nori.Fetcher()

        ls = self.f_list[idx]
        raw_img = Image.open(io.BytesIO(self.f.get(ls[0]))).convert('RGB')
        if self.transform is not None:
            img = self.transform(raw_img)
        else:
            img = raw_img
        raw_img.close()
        label = int(ls[1])

        return idx, img, imagenet_templates[random.randint(0, len(imagenet_templates)-1)](imagenet_classnames[label])

    def __len__(self):
        return len(self.f_list)



def show_tsne(image_backbone_features, image_features, text_features, file_name, title, labels=None):

    logging.info('Fitting T-SNE')
                    
    tsne_img_backbone = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(image_backbone_features)
    tsne_img = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(image_features)
    tsne_text = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(text_features)
    tsne_all = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(torch.cat([image_features, text_features], dim=0))
    
    plt.figure(figsize=(100,25))
    plt.rc('font', size=30) 
    plt.subplots_adjust(top=0.9,wspace=0.05,hspace=0.05)

    plt.subplot(141)
    plt.xticks([])
    plt.yticks([])
    plt.title('image backbone features')
    if labels is None:
        plt.scatter(tsne_img_backbone[:,0], tsne_img_backbone[:,1], s=1.5, c='green', alpha=0.8)
    else:
        plt.scatter(tsne_img_backbone[:,0], tsne_img_backbone[:,1], s=1.5, c=labels, cmap='tab10', alpha=0.8)

    plt.subplot(142)
    plt.xticks([])
    plt.yticks([])
    plt.title('image features')
    if labels is None:
        plt.scatter(tsne_img[:,0], tsne_img[:,1], s=1.5, c='red', alpha=0.8)
    else:
        plt.scatter(tsne_img[:,0], tsne_img[:,1], s=1.5, c=labels, cmap='tab10', alpha=0.8)

    plt.subplot(143)
    plt.xticks([])
    plt.yticks([])
    plt.title('image-text features')
    if labels is None:
        plt.scatter(tsne_all[:len(image_features),0], tsne_all[:len(image_features),1], s=1.5, c='red', alpha=0.5)
        plt.scatter(tsne_all[len(image_features):,0], tsne_all[len(image_features):,1], s=1.5, c='blue', alpha=0.5)
    else:
        plt.scatter(tsne_all[:len(image_features),0], tsne_all[:len(image_features),1], s=1.5, c=labels, cmap='tab10', alpha=0.8)
        plt.scatter(tsne_all[len(image_features):,0], tsne_all[len(image_features):,1], s=1.5, c=labels, cmap='tab10', alpha=0.8)


    plt.subplot(144)
    plt.xticks([])
    plt.yticks([])
    plt.title('text features')
    if labels is None:
        plt.scatter(tsne_text[:,0], tsne_text[:,1], s=1.5, c='blue', alpha=0.8)
    else:
        plt.scatter(tsne_text[:,0], tsne_text[:,1], s=1.5, c=labels, cmap='tab10', alpha=0.8)
    
    plt.suptitle(title)
    plt.savefig(file_name, bbox_inches='tight')

    logging.info(f'T-SNE visuallization saved to: {file_name}')


def extract_feature(student, teacher, dataset_CC, args):
    dataloader_CC = DataLoader(dataset_CC, batch_size=100, num_workers=8, persistent_workers=True)
    
    all_image_backbone_features = []
    all_image_features = []
    all_text_features = []
    for (index, images, texts) in tqdm(dataloader_CC):
        with torch.no_grad():
            text_features = teacher.encode(
                    texts,
                    convert_to_tensor=True, 
                    show_progress_bar=False
                    ).detach()

            raw_image_features = student(images.cuda())    
            image_features = student.text_projection_head(raw_image_features)      
        
        raw_image_features = F.normalize(raw_image_features, dim=1)
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        all_image_backbone_features.append(raw_image_features)
        all_image_features.append(image_features)
        all_text_features.append(text_features)
    
    all_image_backbone_features = torch.stack(all_image_backbone_features).view(-1, all_image_backbone_features[0].size(-1))
    all_image_features = torch.stack(all_image_features).view(-1, all_image_features[0].size(-1))
    all_text_features = torch.stack(all_text_features).view(-1, all_text_features[0].size(-1))

    return all_image_backbone_features.cpu(), all_image_features.cpu(), all_text_features.cpu()


def evaluate_checkpoint(checkpoint_path, epoch, args):
    # load model
    
    logging.info(f'Loading pretrained text trasformer teacher: {args.text_teacher}.')
     
    text_teacher = SentenceTransformer(args.text_teacher).to(device)
    if args.text_teacher in ['clip-ViT-B-32', 'clip-ViT-B-16']:
        args.text_teacher_dim = 512
    else:   
        args.text_teacher_dim = text_teacher.get_sentence_embedding_dimension()

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
    student = add_projection_head(student, student.output_dim, args)
    student = student.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    msg = student.load_state_dict(sd, strict=False)
    logging.info(str(msg))
    logging.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")  
 


    #Conceptual Captions
    dataset_CC = CsvDataset(args.input_filename, preprocess_val, dataset_size=args.num_points)
    image_backbone_features, image_features, text_features = extract_feature(student, text_teacher, dataset_CC, args)
    show_tsne(
        image_backbone_features, image_features, text_features, 
        file_name=os.path.join(args.exp_dir, 'visualization', f'tsne(CC-{len(dataset_CC)})_epoch_{epoch}.png'), 
        title=args.exp_dir)
    
    # ImageNet
    dataset_ImageNet = ImageNet_nori(transform=preprocess_val, split='val')
    image_backbone_features, image_features, text_features  = extract_feature(student, text_teacher, dataset_ImageNet, args)
    print(image_backbone_features.size(), image_features.size(), text_features.size())
    show_tsne(
        image_backbone_features, image_features, text_features, 
        file_name=os.path.join(args.exp_dir, 'visualization', f'tsne(ImageNet-val)_epoch_{epoch}.png'), 
        title=args.exp_dir,
        labels=dataset_ImageNet.labels)
    

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
    num_points = input('Sample how many points for TSNE? (press "enter" to use 200000 points) ')
    single_eval = input('Specify a checkpoint epoch? (press "enter" to scan and evaluate all checkpoints) ')
    
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    params_file = os.path.join(exp_dir, 'params.txt')
    
    args = parse_args()
    args = load_params(params_file, args)

    args.num_points = int(num_points) if num_points else 200000

    args.input_filename = 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv'
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
        
