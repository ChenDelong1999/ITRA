import os
from PIL import Image
import numpy as np
import random
import glob
import torchvision.datasets
from torch.utils.data import Dataset
from data.classname_and_prompt import *

try:
    from refile import smart_open
    import nori2 as nori
    import io
except ImportError:
    # TODO: remove nori dependency when publish codes
    pass

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
AVALIABLE_CLASSIFICATION_DATASETS = [
    'MNIST',
    'CIFAR10',
    'CIFAR100',
    'STL10',
    'FGVCAircraft',
    'StanfordCars',
    'DTD',
    'Food101',
    'Flowers102',
    'OxfordIIITPet',
    'GTSRB',
    'ImageNet',
    'ImageNet-50k',
    'RenderedSST2',
    'EuroSAT',
    'PatternNet',
    'OPTIMAL31',
    'RSC11',
    'AID',
    'MLRSNet',
    'RESISC45',
    'RSICB128',
    'WHURS19',
    'RSICB256',
    'UCMerced',
    'SIRIWHU',
    'RSSCN7'
    ]

def get_dataset(dataset_name, split, root, transform=None, args=None):
    #assert dataset_name in DATASETS
    #assert split in ['train', 'test']

    if dataset_name=='MNIST':
        dataset = torchvision.datasets.MNIST(root, train=(split=='train'), transform=transform, download=True)
        dataset.classes = MNIST.classes
        dataset.templates = MNIST.templates

    elif dataset_name=='CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root, train=(split=='train'), transform=transform)
        dataset.classes = CIFAR10.classes
        dataset.templates = CIFAR10.templates

    elif dataset_name=='CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root, train=(split=='train'), transform=transform, download=True)
        dataset.templates = CIFAR100.templates

    elif dataset_name=='STL10':
        dataset = torchvision.datasets.STL10(root, split=split, transform=transform)
        dataset.templates = STL10.templates

    elif dataset_name=='FGVCAircraft':
        dataset = torchvision.datasets.FGVCAircraft(root, split=split, transform=transform)
        dataset.templates = FGVCAircraft.templates

    elif dataset_name=='StanfordCars':
        dataset = torchvision.datasets.StanfordCars(root, download=True, split=split, transform=transform)
        dataset.templates = StanfordCars.templates

    elif dataset_name=='DTD':
        dataset = torchvision.datasets.DTD(root, download=True, split=split, transform=transform)
        dataset.templates = DescribableTextures.templates

    elif dataset_name=='Food101':
        dataset = torchvision.datasets.Food101(root, download=True, split=split, transform=transform)
        dataset.templates = Food101.templates

    elif dataset_name=='Flowers102':
        dataset = torchvision.datasets.Flowers102(root, download=True, split=split, transform=transform)
        dataset.classes = Flowers102.classes
        dataset.templates = Flowers102.templates

    elif dataset_name=='OxfordIIITPet':
        dataset = torchvision.datasets.OxfordIIITPet(root, download=True, split='trainval' if (split=='train') else 'test', transform=transform)
        dataset.templates = OxfordPets.templates

    elif dataset_name=='GTSRB':
        dataset = torchvision.datasets.GTSRB(root, download=True, split=split, transform=transform)
        dataset.classes = GTSRB.classes
        dataset.templates = GTSRB.templates

    elif dataset_name=='ImageNet':
        split = 'val' if split=='test' else split
        dataset = ImageNet_nori(split=split, transform=transform)
        dataset.templates = ImageNet.templates
        dataset.classes = ImageNet.classes

    elif dataset_name=='ImageNet-CN':
        split = 'val' if split=='test' else split
        dataset = ImageNet_nori(split=split, transform=transform)
        dataset.templates = ImageNet_CN.templates
        dataset.classes = ImageNet_CN.classes

    elif dataset_name=='ImageNet-50k':
        dataset = ImageNet_50k(transform=transform) if split=='train' else ImageNet_nori(split='val', transform=transform)
        dataset.templates = ImageNet.templates
        dataset.classes = ImageNet.classes

    elif dataset_name=='RenderedSST2':
        dataset = torchvision.datasets.RenderedSST2(root, download=True, split=split, transform=transform)
        dataset.classes = SST2.classes
        dataset.templates = SST2.templates

    elif dataset_name == 'EuroSAT':
        cur_dataset_root = f"{root}/EuroSAT/2750"
        # 划分数据集
        train_paths, test_paths = data_split(cur_dataset_root, args)
        # class_names 对应文本提示的类别名
        # file_class_names 对应数据文件的类别名
        class_names = EuroSAT.classes
        file_class_names = EuroSAT.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        # 获取当前数据集的类别名以及文本提示模板
        dataset.classes = class_names
        dataset.templates = EuroSAT.templates

    elif dataset_name == 'PatternNet':
        cur_dataset_root = f"{root}/PatternNet/images"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = PatternNet.classes
        file_class_names = PatternNet.file_class_names

        if split=='train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split=='test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = PatternNet.templates

    elif dataset_name == 'OPTIMAL31':
        cur_dataset_root = f"{root}/OPTIMAL-31/Images"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = OPTIMAL31.classes
        file_class_names = OPTIMAL31.file_class_names

        if split=='train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split=='test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = OPTIMAL31.templates

    elif dataset_name == 'RSC11':
        cur_dataset_root = f"{root}/RS_C11_Database"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = RSC11.classes
        file_class_names = RSC11.file_class_names

        if split=='train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split=='test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = RSC11.templates

    elif dataset_name == 'AID':
        cur_dataset_root = f"{root}/AID_dataset/AID"
        # 划分数据集
        train_paths, test_paths = data_split(cur_dataset_root, args)
        # class_names 对应文本提示的类别名
        # file_class_names 对应数据文件的类别名
        class_names = AID.classes
        file_class_names = AID.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        # 获取当前数据集的类别名以及文本提示模板
        dataset.classes = class_names
        dataset.templates = AID.templates

    elif dataset_name == 'MLRSNet':
        cur_dataset_root = f"{root}/MLRSNet/Images"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = MLRSNet.classes
        file_class_names = MLRSNet.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = MLRSNet.templates

    elif dataset_name == 'RESISC45':
        cur_dataset_root = f"{root}/NWPU-RESISC45"
        # 划分数据集
        train_paths, test_paths = data_split(cur_dataset_root, args)
        # class_names 对应文本提示的类别名
        # file_class_names 对应数据文件的类别名
        class_names = RESISC45.classes
        file_class_names = RESISC45.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        # 获取当前数据集的类别名以及文本提示模板
        dataset.classes = class_names
        dataset.templates = RESISC45.templates

    elif dataset_name == 'RSICB128':
        cur_dataset_root = f"{root}/RSI-CB128"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = RSICB128.classes
        file_class_names = RSICB128.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = RSICB128.templates

    elif dataset_name == 'WHURS19':
        cur_dataset_root = f"{root}/WHU-RS19"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = WHURS19.classes
        file_class_names = WHURS19.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = WHURS19.templates

    elif dataset_name == 'RSICB256':
        cur_dataset_root = f"{root}/RSI-CB256"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = RSICB256.classes
        file_class_names = RSICB256.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = RSICB256.templates

    elif dataset_name == 'UCMerced':
        cur_dataset_root = f"{root}/UCMerced_LandUse/Images"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = UCMerced.classes
        file_class_names = UCMerced.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = UCMerced.templates

    elif dataset_name == 'WHUearth':
        cur_dataset_root = f"{root}/Google dataset of SIRI-WHU_earth_im_tiff/12class_tif"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = WHUearth.classes
        file_class_names = WHUearth.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = WHUearth.templates

    elif dataset_name == 'RS2800':
        cur_dataset_root = f"{root}/RS2800/RS_images_2800"
        train_paths, test_paths = data_split(cur_dataset_root, args)
        class_names = RS2800.classes
        file_class_names = RS2800.file_class_names

        if split == 'train':
            dataset = remote_sensing_dataset(paths=train_paths, transform=transform,
                                             file_class_names=file_class_names)
        elif split == 'test':
            dataset = remote_sensing_dataset(paths=test_paths, transform=transform,
                                             file_class_names=file_class_names)

        dataset.classes = class_names
        dataset.templates = RS2800.templates

    '''
    elif dataset_name=='YourCustomDataset':
        dataset = your custome dataset (torch.utils.data.Dataset)
        dataset.classes = a list of classnames
        dataset.templates = a list of prompt templemts
        # remember to add your 'YourCustomDataset' to AVALIABLE_CLASSIFICATION_DATASETS
    '''

    dataset.classes = [dataset.classes[i].replace('_', ' ') for i in range(len(dataset.classes))]
    dataset.classes = [dataset.classes[i].replace('/', ' ') for i in range(len(dataset.classes))]

    return dataset


def data_split(data_root, args):
    train_paths, valid_paths, test_paths = [], [], []
    # 设置训练集和测试集的比例
    few_shot_train_paths = []
    train_scale, test_scale = 0.8, 0.2

    few_shot = 0 if args.linear_prob_setting == "8:2" else int(args.linear_prob_setting.split('_')[0])

    for folder in [os.path.basename(folder) for folder in sorted(glob.glob(os.path.join(data_root, "*")))]:
        keep_paths = glob.glob(os.path.join(data_root, folder, "*"))

        current_data_length = len(keep_paths)
        train_stop = int(current_data_length * train_scale)

        train_paths.extend(keep_paths[:train_stop])
        few_shot_train_paths.extend(random.sample(train_paths,few_shot))
        test_paths.extend(keep_paths[train_stop:])

    if few_shot > 0:
        print(f"few_shot_train_paths_len:{len(few_shot_train_paths)}, test_paths_len:{len(test_paths)}")
        return few_shot_train_paths, test_paths

    else:
        print(f"train_paths_len:{len(train_paths)}, test_paths_len:{len(test_paths)}")
        return train_paths, test_paths


class remote_sensing_dataset(Dataset):
    def __init__(self, paths, transform, file_class_names):
        self.transform = transform
        self.paths = paths
        self.class_names = file_class_names

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        class_name = self.paths[index].split('/')[-2]
        # class_name = class_name.replace('_', ' ')
        return img, self.class_names.index(class_name)

    def __len__(self):
        return len(self.paths)


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





if __name__=='__main__':
    # for dataset_name in AVALIABLE_CLASSIFICATION_DATASETS:
    #     print('='*64)
    #     for split in ['test', 'train']:
    #         dataset = get_dataset(dataset_name, split)
    #         print(dataset)
    #         if split=='test':
    #             print(dataset.classes)
    #             print(len(dataset.classes))
    #             print(dataset.templates)
    #             print(len(dataset.templates))
    dataset = get_dataset('EuroSAT', 'train', '/datasets/remote_sensing')
    print(dataset[0])
    print(dataset.templates)