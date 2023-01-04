import json
import torch
from refile import smart_open
import nori2 as nori
import io
from PIL import Image
import pandas as pd
import numpy as np
from training.evaluations.openai_templets.ImageNet import classes as imagenet_classnames

class ImageNetCaptions():

    def __init__(self) -> None:
        dataset_file = '/data/Datasets/imagenet_captions.json'
        with open(dataset_file,'r') as load_f:
            dataset_dict = json.load(load_f)
        self.images = []
        self.captions = []

        for sample in dataset_dict:
            # 'filename', 'title', 'tags', 'description', 'wnid'
            self.images.append(sample['filename'])
            tags = ' '.join(sample['tags'])
            self.captions.append(' '.join([sample['title'], tags, sample['description']]).replace('\n',' ').replace('\t',' '))


class ImageNet_nori(torch.utils.data.Dataset):
    
    def __init__(self, imagenet_captions, transform, split='val'):

        super(ImageNet_nori, self).__init__()
        if split=='train':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.train.nori.list"
            #nori_path = "s3://public-datasets-contrib/ILSVRC2012/processed/nori/imagenet.train.nori.list"
        elif split=='val':
            nori_path = "s3://generalDetection/ILSVRC2012/imagenet.val.nori.list"
            #nori_path = "s3://public-datasets-contrib/ILSVRC2012/processed/nori/imagenet.val.nori.list"

        self.f = None #nori.Fetcher()
        self.f_list = []
        self.file_names = []
        self.labels = []
        self.transform = transform

        self.imagenet_captions = imagenet_captions
        with smart_open(nori_path, "r") as f:
            for line in f:
                line = line.strip().split()
                file_name = line[2].split('/')[1]
                self.f_list.append(line)
                self.file_names.append(file_name)
                self.labels.append(int(line[1]))
    

    def __getitem__(self, idx):
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
        return len(self.f_list)



if __name__=='__main__':
    imagenet_captions = ImageNetCaptions()
    imagenet = ImageNet_nori(
        imagenet_captions=imagenet_captions, 
        transform=None, 
        split='train'
        )

    print(imagenet.file_names[:500])
    print(len(imagenet.file_names))

    print(imagenet.labels[:500])
    print(len(imagenet.labels))

    file_name_to_label = {}
    for i in range(len(imagenet.file_names)):
        file_name_to_label[imagenet.file_names[i]] = imagenet.labels[i]

    print(imagenet_captions.images[:500])
    print(len(imagenet_captions.images))

    imagenet_captions.labels = []
    for i in range(len(imagenet_captions.images)):
        imagenet_captions.labels.append(file_name_to_label[imagenet_captions.images[i]])

    print(imagenet_captions.labels[:500])
    print(len(imagenet_captions.labels))

    imagenet_classnames = np.array(imagenet_classnames)

    df = pd.DataFrame(data={
        'label':imagenet_captions.labels,
        'class_name':imagenet_classnames[imagenet_captions.labels],
        'file_name':imagenet_captions.images,
        'caption':imagenet_captions.captions,
    })
    print(df)
    df.to_csv('data/ImageNet_captions_labeled.csv',index=False, sep='\t')

    