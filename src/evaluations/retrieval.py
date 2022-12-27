import torch
import logging
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from clip_benchmark.datasets.builder import get_dataset_collate_fn

try:
    from refile import smart_open
    import nori2 as nori
    import io
except ImportError:
    # TODO: remove nori dependency when publish codes
    pass


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, aug=None, sep="\t", nori_dataset=False, images_dir=''):
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

        self.duplicate()       

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        texts = self.captions[index]
        #images = self.transforms(Image.open(str(self.images[index])))
        if self.nori_dataset:
            if self.f is None:
                self.f = nori.Fetcher()
            image = Image.open(io.BytesIO(self.f.get(self.images[index])))
        else:
            image = Image.open(os.path.join(self.images_dir, str(self.images[index])))
        
        image = self.transforms(image)
        
        return image, texts

    def duplicate(self):
        unique_images, indexs = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            logging.debug(f'Amoung all {len(self.images)} images, there are only {len(unique_images)} unique images. Dupication will be performed to enable one-image-to-multiple-text retrieval.')

            self.duplicated_images = []
            self.duplicated_captions = []
            for index in indexs:
                self.duplicated_images.append(self.images[index])
                same_indexs = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = []
                for same_index in same_indexs:
                    captions.append(self.captions[same_index])
                self.duplicated_captions.append(captions)

            self.images = self.duplicated_images
            self.captions = self.duplicated_captions    
        

def retrieval_evaluation(model, epoch, preprocess, args, recall_k_list=[1,5,10]):
 
    """
    Modified from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda
    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """

    if args.retrieval_frequency == 0:
        return {}
    if (epoch % args.retrieval_frequency) != 0 and epoch != args.epochs:
        return {}

    
    if args.retrieval_data=='mscoco_captions_2014':
        from torchvision.datasets.coco import CocoCaptions
        coco_val_root = os.path.join(args.eval_data_dir, 'coco2014/val2014')
        coco_val_json = os.path.join(args.eval_data_dir, 'coco2014/annotations/captions_val2014.json')
        dataset = CocoCaptions(root=coco_val_root, annFile=coco_val_json, transform=preprocess)
        
    elif args.retrieval_data=='mscoco_captions':
        from torchvision.datasets.coco import CocoCaptions
        coco_val_root = os.path.join(args.eval_data_dir, 'coco2017/val2017')
        coco_val_json = os.path.join(args.eval_data_dir, 'coco2017/annotations/captions_val2017.json')
        dataset = CocoCaptions(root=coco_val_root, annFile=coco_val_json, transform=preprocess)
    else:        
        dataset = CsvDataset(
            input_filename=args.retrieval_data,
            transforms=preprocess,
            img_key=args.retrieval_csv_img_key,
            caption_key=args.retrieval_csv_caption_key,
            sep=args.retrieval_csv_separator,
            nori_dataset=args.retrieval_nori_dataset,
            images_dir=os.path.join(args.eval_data_dir, args.retrieval_images_dir)
        )


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=get_dataset_collate_fn('mscoco_captions')
    )
    n_batches = len(dataloader)

    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    
    for batch_images, batch_texts, inds in tqdm(dataloader, total=n_batches): 
        batch_images = batch_images.to(args.device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # tokenize all texts in the batch
        batch_texts = [text for i, texts in enumerate(batch_texts) for text in texts]
        
        # compute the embedding of images and texts
        with torch.no_grad():
            if args.distributed and not args.horovod:
                batch_image_features = model.module.encode_image(batch_images, projection=True)
                batch_text_features = model.module.encode_text(batch_texts, projection=True)
            else:
                batch_image_features = model.encode_image(batch_images, projection=True)
                batch_text_features = model.encode_text(batch_texts, projection=True)

            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        '''
        Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        for each image, that number will be greater than 1 for text retrieval.
        However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        it over the dataset.
        '''
        metrics[f"retrieval-image2text-R@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device, k=recall_k)>0).float().mean().item() * 100
        
    for recall_k in recall_k_list:
        metrics[f"retrieval-text2image-R@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, args.device, k=recall_k)>0).float().mean().item() * 100

    metrics[f"retrieval-mean-recall"] = np.mean(list(metrics.values()))
    
    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)


    return metrics
    

