
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from evaluations.downstream_datasets import get_dataset, AVALIABLE_DATASETS


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


try:
    from refile import smart_open
    import nori2 as nori
    import io
except ImportError:
    # TODO: remove nori dependency when publish codes
    pass


from torchvision.datasets.coco import CocoCaptions
import os
import random


class ItraDataset(Dataset):
    def __init__(self, datasets, args, index_mapping=None) -> None:
        super().__init__()
        self.datasets = datasets
        
        self.n_samples = []
        self.index2dataset = []
        self.n_classes = []

        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            self.n_samples.append(len(dataset))
            self.index2dataset.extend([i]*len(dataset))

            if dataset.type=='classification':
                self.n_classes.append(len(dataset.base_dataset.classes))
            elif dataset.type=='image-text-pairs':
                self.n_classes.append(len(dataset))

        if len(datasets) > 0:
            logging.info(f'Building {len(datasets)} dataset: [{args.train_data}]. n-samples: {self.n_samples}, n-classes: {self.n_classes}')
        
        if index_mapping is None:
            self.index_mapping=torch.arange(sum(self.n_samples))
        else:
            self.index_mapping = index_mapping


    def __len__(self):
        return sum(self.n_samples)
    
    def __getitem__(self, episodic_index):
        
        index = self.index_mapping[episodic_index]
        dataset_index = self.index2dataset[index]
        
        previous_samples_sum = sum(self.n_samples[:dataset_index])         
        subset_index, image, caption, label = self.datasets[dataset_index][index-previous_samples_sum]

        if label!=-1:
            previous_classes_sum = sum(self.n_classes[:dataset_index])  
            label += previous_classes_sum

        return index, image, caption, label
        

class COCOCaptionsDataset(Dataset):
    def __init__(self, dataset_name, transforms, args):
        
        if dataset_name=='mscoco_captions_2014':
            coco_train_root = os.path.join(args.eval_data_dir, 'coco2014/train2014')
            coco_train_json = os.path.join(args.eval_data_dir, 'coco2014/annotations/captions_train2014.json')
            self.coco_dataset = CocoCaptions(root=coco_train_root, annFile=coco_train_json, transform=transforms)
            
        elif dataset_name=='mscoco_captions':
            coco_train_root = os.path.join(args.eval_data_dir, 'coco2017/train2017')
            coco_train_json = os.path.join(args.eval_data_dir, 'coco2017/annotations/captions_train2017.json')
            self.coco_dataset = CocoCaptions(root=coco_train_root, annFile=coco_train_json, transform=transforms)
        
        else:
            raise RuntimeError(f'{dataset_name} not supported!')
        
        self.type = 'image-text-pairs'
        
    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        img, captions = self.coco_dataset[index]
        return index, img, captions[random.randint(0,4)], -1
  

class PromptedClassificationDataset(Dataset):
    def __init__(self, dataset_name, transforms, args) -> None:
        super().__init__()
        logging.debug(f'Building classification dataset {dataset_name} with prompt.')
        self.base_dataset = get_dataset(dataset_name, split='train', root=args.eval_data_dir, transform=transforms)
        self.type = 'classification'
    
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        class_name = self.base_dataset.classes[label]
        template = self.base_dataset.templates[random.randint(0,len(self.base_dataset.templates)-1)]
        caption = template.replace('{}',class_name)
        return index, image, caption, label


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, aug=None, sep="\t", dataset_size=None, index_mapping=None, skip_image=False, nori_dataset=False, images_dir=''):
        logging.debug(f'Loading csv data from {input_filename}.')
        if input_filename[:2]=='s3':
            df = pd.read_csv(smart_open(input_filename, "r"), sep=sep)
        elif 'rsicd' in input_filename:
            df = pd.read_csv(input_filename, sep=sep, encoding='gb18030')
        else:
            df = pd.read_csv(input_filename, sep=sep)
        
        self.nori_dataset = nori_dataset
        self.f = None
        self.images_dir = images_dir

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        self.transforms = transforms
        self.aug = aug
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
        
        self.skip_image = skip_image
        self.type = 'image-text-pairs'
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        texts = str(self.captions[index].decode('utf-8'))

        if self.skip_image:
            images = texts # skip image forward for efficient teacher caching 
        else:
            #images = self.transforms(Image.open(str(self.images[index])))
            if self.nori_dataset:
                if self.f is None:
                    self.f = nori.Fetcher()
                image = Image.open(io.BytesIO(self.f.get(self.images[index].decode('utf-8'))))
            else:
                image = Image.open(os.path.join(self.images_dir, str(self.images[index].decode('utf-8'))))
            
            image_train = self.transforms(image)
            if self.aug is not None:
                images = (image_train, self.aug(image))
            else:
                images = image_train
        
        return index, images, texts, -1

    def get_data(self, episode_index):
        idx = self.index_mapping[episode_index]
            
        # get image data
        if self.nori_dataset:
            if self.f is None:
                self.f = nori.Fetcher()
            pic = Image.open(io.BytesIO(self.f.get(self.images[idx].decode('utf-8'))))
        else:
            pic = Image.open(str(self.images[idx].decode('utf-8')))
        
        
        image = self.inversed_normalize(self.transforms(pic))
        texts = self.captions[idx].decode('utf-8')
        
        return image, texts

# TODO: remove nori when publish
class ImageNet_nori(Dataset):
    
    def __init__(self, transform, split='val'):

        super(ImageNet_nori, self).__init__()
        if split=='train':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.train.nori.list"
            #nori_path = "s3://public-datasets-contrib/ILSVRC2012/processed/nori/imagenet.train.nori.list"
        elif split=='val':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.val.nori.list"
            #nori_path = "s3://public-datasets-contrib/ILSVRC2012/processed/nori/imagenet.val.nori.list"

        self.f = None #nori.Fetcher()
        self.f_list = []
        self.transform = transform

        with smart_open(nori_path, "r") as f:
            for line in f:
                self.f_list.append(line.strip().split())
              
    def __getitem__(self, idx):
        if self.f is None:
            self.f = nori.Fetcher()

        ls = self.f_list[idx]
        raw_img = Image.open(io.BytesIO(self.f.get(ls[0])))
        if self.transform is not None:
            img = self.transform(raw_img)
            raw_img.close()
        else:
            img = raw_img
        label = int(ls[1])

        return img, label

    def __len__(self):
        return len(self.f_list)

class ImageNet_50k(Dataset):
    # modified from https://git-core.megvii-inc.com/lizeming/mt/-/blob/dev/megssl/data/datasets/imagenet.py
    def __init__(self, transform):

        super(ImageNet_50k, self).__init__()
        nori_path = "s3://generalDetection/ILSVRC2012/imagenet.train.nori.list"

        self.f = None #nori.Fetcher()
        self.f_list = []
        self.transform = transform

        with smart_open(nori_path, "r") as f:
            for line in f:
                self.f_list.append(line.strip().split())
              
    def __getitem__(self, idx):
        idx = int(idx * 25)
        if self.f is None:
            self.f = nori.Fetcher()

        ls = self.f_list[idx]
        raw_img = Image.open(io.BytesIO(self.f.get(ls[0])))
        if self.transform is not None:
            img = self.transform(raw_img)
        else:
            img = raw_img
        raw_img.close()
        label = int(ls[1])

        return img, label

    def __len__(self):
        return 50000

@dataclass
class DataInfo:
    dataset: Dataset
    dataloader: DataLoader
    sampler: DistributedSampler


def get_csv_dataset(args, preprocess_fn, aug, is_train, index_mapping):

    datasets = []
    for dataset_name in args.train_data.split(','):
        if 'mscoco_captions' in dataset_name:
            dataset = COCOCaptionsDataset(dataset_name=dataset_name, transforms=preprocess_fn, args=args)
        elif dataset_name in AVALIABLE_DATASETS:
            dataset = PromptedClassificationDataset(dataset_name=dataset_name, transforms=preprocess_fn, args=args)
        else:
            dataset = CsvDataset(
                dataset_name,
                preprocess_fn,
                aug=aug if args.BYOL else None,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                dataset_size=args.dataset_size,
                #skip_image=args.cache_teacher is not None
                nori_dataset=args.nori_dataset,
                images_dir=args.images_dir
                )
        datasets.append(dataset)

    dataset = ItraDataset(datasets, args, index_mapping)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=False,
        persistent_workers=args.workers>0
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataset, dataloader, sampler)



def get_data(args, preprocess_fns, index_mapping):
    preprocess_train, preprocess_val, preprocess_aug = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_csv_dataset(args, preprocess_train, aug=preprocess_aug,  is_train=True, index_mapping=index_mapping)

    return data
