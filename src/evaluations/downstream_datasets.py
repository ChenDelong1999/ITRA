import os
from PIL import Image
import numpy as np
import random
import glob
import torchvision.datasets
from training.data import ImageNet_nori, ImageNet_50k
from evaluations.openai_templets import *

AVALIABLE_DATASETS = [
    'ImageNet-CN', 
    'MNIST', 
    'CIFAR10', 
    'CIFAR100', 
    'STL10', 
    'SUN397', 
    'FGVCAircraft', 
    'StanfordCars', 
    'Caltech101', 
    'DTD', 
    'Food101', 
    'Flowers102', 
    'OxfordIIITPet', 
    #'GTSRB',
    'ImageNet', 
    'ImageNet-50k',
    'EuroSAT',
    'RenderedSST2',
    'CLEVER'
    ]

def get_dataset(dataset_name, split, root='/data/Datasets', transform=None):
    #assert dataset_name in DATASETS
    #assert split in ['train', 'test']

    if dataset_name=='MNIST':
        dataset = torchvision.datasets.MNIST(root, train=(split=='train'), transform=transform)
        dataset.classes = MNIST.classes
        dataset.templates = MNIST.templates
    
    elif dataset_name=='CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root, train=(split=='train'), transform=transform)
        dataset.classes = CIFAR10.classes
        dataset.templates = ImageNet_CN.templates
        # dataset.templates = CIFAR10.templates
    
    elif dataset_name=='CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root, train=(split=='train'), transform=transform)
        dataset.templates = CIFAR100.templates
    
    elif dataset_name=='STL10':
        dataset = torchvision.datasets.STL10(root, split=split, transform=transform)
        dataset.templates = STL10.templates
    
    elif dataset_name=='SUN397':
        dataset = torchvision.datasets.SUN397(root, transform=transform)
        dataset.templates = SUN397.templates
        partition = 'Training_01.txt' if split=='train' else 'Testing_01.txt'
        files = open(os.path.join(root, 'SUN397' ,partition)).read().splitlines()
        _image_files = []
        for i in range(len(files)):
            _image_files.append(root + '/SUN397'+ files[i])
        _labels = [dataset.class_to_idx["/".join(path.relative_to(dataset._data_dir).parts[1:-1])] for path in dataset._image_files]
        dataset._image_files = _image_files
        dataset._labels = _labels
    
    elif dataset_name=='FGVCAircraft':
        dataset = torchvision.datasets.FGVCAircraft(root, split=split, transform=transform)
        dataset.templates = FGVCAircraft.templates
    
    elif dataset_name=='StanfordCars':
        dataset = torchvision.datasets.StanfordCars(root, download=True, split=split, transform=transform)
        dataset.templates = StanfordCars.templates
    
    elif dataset_name=='Caltech101':
        # TODO: dataset spliting
        dataset = torchvision.datasets.Caltech101(root, transform=transform)
        dataset.classes = Caltech101.classes
        dataset.templates = Caltech101.templates
    
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
        dataset = torchvision.datasets.GTSRB(root, download=False, split=split, transform=transform)
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

    elif dataset_name=='EuroSAT':
        #dataset = torchvision.datasets.EuroSAT(root, download=True,  transform=transform)
        # if split=='test':
        #     split = 'val'
        # dataset = torchvision.datasets.ImageFolder(os.path.join(root, 'eurosat', split), transform=transform)

        # https://github.com/openai/CLIP/issues/45#issuecomment-926334608
        EuroSAT_root = f"{root}/eurosat/2750"
        seed = 42
        random.seed(seed)
        train_paths, valid_paths, test_paths = [], [], []
        for folder in [os.path.basename(folder) for folder in sorted(glob.glob(os.path.join(EuroSAT_root, "*")))]:
            keep_paths = random.sample(glob.glob(os.path.join(EuroSAT_root, folder, "*")), 1500)
            #keep_paths = [os.path.relpath(path, EuroSAT_root) for path in keep_paths]
            train_paths.extend(keep_paths[:1000])
            #valid_paths.extend(keep_paths[500:1000])
            test_paths.extend(keep_paths[1000:])

        class PathDataset():
            def __init__(self, paths, transform):
                self.transform = transform
                self.paths = paths
                self.class_names = ['River', 'AnnualCrop', 'HerbaceousVegetation', 'Industrial', 'Residential', 'Highway', 'Pasture', 'Forest', 'SeaLake', 'PermanentCrop']

            def __getitem__(self, index):
                img = Image.open(self.paths[index])
                if self.transform is not None:
                    img = self.transform(img)
                class_name = self.paths[index].split('/')[-1]
                class_name = class_name.split('_')[0]
                return img, self.class_names.index(class_name)

            def __len__(self):
                return len(self.paths)
        if split=='train':
            dataset = PathDataset(paths=train_paths, transform=transform)
        elif split=='test':
            dataset = PathDataset(paths=test_paths, transform=transform)
        dataset.classes = EuroSAT.classes
        dataset.templates = EuroSAT.templates

    elif dataset_name=='RenderedSST2':
        dataset = torchvision.datasets.RenderedSST2(root, download=True, split=split, transform=transform)
        dataset.classes = SST2.classes
        dataset.templates = SST2.templates

    elif dataset_name=='CLEVER':
        
        class CLEVERDataset():
            def __init__(self, root, split, transform):
                self.transform = transform
                _images = np.load(os.path.join(root, 'clevr_count' , split+'_images.npy'))
                _labels = np.load(os.path.join(root, 'clevr_count' , split+'_labels.npy'))
                self.images = []
                self.labels = []

                for i in range(len(_labels)):
                    self.images.append(str(_images[i]))
                    self.labels.append(int(_labels[i].replace('count_', '')))
                    
            def __getitem__(self, index):
                img, target = self.images[index], int(self.labels[index])
                img = Image.open(img)
                if self.transform is not None:
                    img = self.transform(img)
                return img, target

            def __len__(self):
                return len(self.images)
        

        if split=='test':
            split = 'val'
        dataset = CLEVERDataset(root, split, transform)
        dataset.classes = CLEVERCounts.classes
        dataset.templates = CLEVERCounts.templates

    
    #zeroshot_datasets = ['imagenet', 'cifar10', 'cifar100', 'stl10', 'birdsnap','country211', 'flowers102', 'gtsrb', 'ucf101','stanford_cars']
    # for ['birdsnap', 'country211', 'flowers102', 'gtsrb', 'stanford_cars', 'ucf101']
    elif  dataset_name in ['birdsnap', 'country211', 'flowers102', 'ucf101']:
        assert split == 'test'
        data_path = f'{root}/{dataset_name}/test'
        if dataset_name == 'ucf101':
            data_path += 'list01'
        dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)

        if dataset_name=='birdsnap':
            dataset.classes = Birdsnap.classes
            dataset.templates = Birdsnap.templates
            # empty_indexs = [46, 66, 123, 299, 302, 351, 403, 436, 465]
            # for empty_index in empty_indexs[::-1]:
            #     del dataset.classes[empty_index]

        elif dataset_name=='country211':
            dataset.classes = Country211.classes
            dataset.templates = Country211.templates

        elif dataset_name=='flowers102':
            dataset.classes = Flowers102.classes
            dataset.templates = Flowers102.templates

        elif dataset_name=='ucf101':
            dataset.classes = UCF101.classes
            dataset.templates = UCF101.templates

        
    
    dataset.classes = [dataset.classes[i].replace('_', ' ') for i in range(len(dataset.classes))]
    dataset.classes = [dataset.classes[i].replace('/', ' ') for i in range(len(dataset.classes))]
    return dataset

if __name__=='__main__':
    #for dataset_name in AVALIABLE_DATASETS:
    for dataset_name in ['ImageNet-CN']:
        print('='*64)
        for split in ['test', 'train']:
            dataset = get_dataset(dataset_name, split)
            print(dataset)
            if split=='test':
                print(dataset.classes)
                print(len(dataset.classes))
                print(dataset.templates)
                print(len(dataset.templates))