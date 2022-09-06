
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, aug=None, sep="\t", dataset_size=None, index_mapping=None, skip_image=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        if input_filename[:2]=='s3':
            self.using_nori = True
            df = pd.read_csv(smart_open(input_filename, "r"), sep=sep)
            self.f = None
        else:
            #self.using_nori = False
            self.using_nori = True
            self.f = None
            df = pd.read_csv(input_filename, sep=sep)

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
        
        if index_mapping is None:
            self.index_mapping=torch.arange(len(self.captions))
        else:
            self.index_mapping = index_mapping
        
        self.skip_image = skip_image
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, episodic_index):
        index = self.index_mapping[episodic_index]
        texts = str(self.captions[index].decode('utf-8'))

        if self.skip_image:
            images = texts # skip image forward for efficient teacher caching 
        else:
            #images = self.transforms(Image.open(str(self.images[index])))
            if self.using_nori:
                if self.f is None:
                    self.f = nori.Fetcher()
                image = Image.open(io.BytesIO(self.f.get(self.images[index].decode('utf-8'))))
            else:
                image = Image.open(str(self.images[index].decode('utf-8')))
            
            image_train = self.transforms(image)
            if self.aug is not None:
                images = (image_train, self.aug(image))
            else:
                images = image_train
        
        return episodic_index, images, texts #[:100]# FIXME: '[:100]' is a temperate solution of CLIP's tokenizer overlength bug
    
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

# TODO: remove nori when publish
class ImageNet_nori(Dataset):
    # modified from https://git-core.megvii-inc.com/lizeming/mt/-/blob/dev/megssl/data/datasets/imagenet.py
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
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        aug=aug if args.BYOL else None,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        dataset_size=args.dataset_size,
        index_mapping=index_mapping,
        skip_image=args.cache_teacher is not None
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        persistent_workers=True
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
