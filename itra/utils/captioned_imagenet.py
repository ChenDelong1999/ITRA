
import sys
import os
import numpy as np
import pandas as pd
from data.classification_datasets import get_dataset


class CaptionedImageNet():
    def __init__(self, path, split,preprocess) -> None:
        if split=='val':
            self.caption_dataset = pd.read_csv(f'{path}/captioned-ImageNet-test.csv')  
        elif split=='train':
            self.caption_dataset = pd.read_csv(f'{path}/captioned-ImageNet-train-rank_0.csv')  
            for i in range(1,8):    
                self.caption_dataset = pd.concat([self.caption_dataset, pd.read_csv(f'{path}/captioned-ImageNet-train-rank_{i}.csv')])  
        #print(self.caption_dataset)
        self.captions = list(self.caption_dataset['caption'].astype(str).values)
        self.indexs = list(self.caption_dataset['index'].values)

        self.image_dataset = get_dataset(dataset_name='ImageNet', split=split, root='/data/Datasets', transform=preprocess)
        labels = np.array(self.image_dataset.f_list)[:,1].astype(int)
        self.labels = list(labels[self.indexs])

        for i in range(len(self.captions)):
            self.captions[i] = self.captions[i][:120]
        
    
    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        #image, label = self.image_dataset[index]
        #label = int(self.image_dataset.f_list[index][1])
        label = self.labels[index]
        caption = self.caption_dataset.loc[self.caption_dataset['index'] == index]['caption'].values[0]
        class_name = self.caption_dataset.loc[self.caption_dataset['index'] == index]['class_name'].values[0]
        return label, caption, class_name





if __name__=='__main__':
    dataset = CaptionedImageNet(path='data/captioned_imagenet', split='train', preprocess=None)
    
    for i in range(len(dataset)):
        label, caption, class_name = dataset[i]
        print(i, label, class_name, caption)
    
    
    exit()

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    rank = int(sys.argv[1])
    world_size = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    print(f'rank {rank} of {world_size}')

    dataset_name = 'ImageNet'
    split = 'train'

    root = '/data/Datasets'
    preprocess = None
    dataset = get_dataset(dataset_name=dataset_name, split=split, root=root, transform=preprocess)
    img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')

    data_frame = None
    for i in range(len(dataset)):
        if i%world_size==rank:
            img, label = dataset[i]
            class_name = dataset.classes[label]
            caption = img_captioning({'image': img})['caption']
            print(f"{i}/{len(dataset)}({int(i/len(dataset)*100)}%) [class name] {class_name}\t[caption] {caption}")

            current = pd.DataFrame({
                'index':[i],
                'class_name':[class_name],
                'caption':[caption]
            })
            if data_frame is None:
                data_frame = current
            else:
                data_frame = pd.concat([data_frame, current])
        
        if i%100==0 and data_frame is not None:
            data_frame.to_csv(f'data/captioned-{dataset_name}-{split}-rank_{rank}.csv', index=False)

    data_frame.to_csv(f'data/captioned-{dataset_name}-{split}-rank_{rank}.csv', index=False)
