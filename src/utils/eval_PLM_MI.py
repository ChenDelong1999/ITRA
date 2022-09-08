
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from torch import nn
from training.evaluations.coco_retrieval import CocoCaptions


from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from mutual_information.transrate import transrate
from mutual_information.logme import log_maximum_evidence
from mutual_information.hscore import h_score

from openTSNE import TSNE
from utils.captioned_imagenet import CaptionedImageNet
import matplotlib.pyplot as plt


def get_coco(split='val'):
    
    eval_data_dir = '/data/Datasets'
    coco_val_root = os.path.join(eval_data_dir, 'coco2017/val2017')
    coco_val_json = os.path.join(eval_data_dir, 'coco2017/annotations/captions_val2017.json')
    coco_train_root = os.path.join(eval_data_dir, 'coco2017/train2017')
    coco_train_json = os.path.join(eval_data_dir, 'coco2017/annotations/captions_train2017.json')

    if split=='val':
        coco_dataset = CocoCaptions(root=coco_val_root, annFile=coco_val_json)
    elif split=='train':
        coco_dataset = CocoCaptions(root=coco_train_root, annFile=coco_train_json)

    all_labels = []
    all_captions = []

    for i in range(len(coco_dataset)):
        texts = coco_dataset._load_target(coco_dataset.ids[i])
        all_labels.extend([i]*len(texts))
        all_captions.extend(texts)
    
    all_labels = np.array(all_labels)

    return all_captions, all_labels
        

def get_imagenet_captions():
    df = pd.read_csv('data/ImageNet_captions_labeled.csv', sep='\t',  lineterminator='\n')
    all_labels = np.array(df['label'].values.tolist())
    all_captions = df['caption'].values.tolist()

    return all_captions, all_labels


def get_captioned_imagenet(split='val'):
    dataset = CaptionedImageNet(path='data/captioned_imagenet', split=split, preprocess=None)
    all_labels = np.array(dataset.labels)
    all_captions = dataset.captions

    return all_captions, all_labels


def save_tsne(features, labels, file_name):
    tsne = TSNE(verbose=True, n_jobs=64, n_iter=500).fit(features)
    plt.figure(figsize=(25,25))
    plt.rc('font', size=30) 
    plt.xticks([])
    plt.yticks([])
    plt.title(file_name)
    plt.scatter(tsne[:,0], tsne[:,1], s=1.5, c=labels, cmap='hsv', alpha=0.8)
    plt.savefig(file_name, bbox_inches='tight')


class WrappedHuggingfaceTransformer(nn.Module):
    def __init__(self, huggingface_transformer) -> None:
        super().__init__()        
        config = AutoConfig.from_pretrained(huggingface_transformer)
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_transformer)
        self.text_backbone = AutoModel.from_pretrained(huggingface_transformer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_model_builder='huggingface-transformer'
        self.device = 'cuda'
        
    # sentence-transformers API
    def encode(self, sentences, batch_size=32, show_progress_bar=None, convert_to_numpy=True, convert_to_tensor=True):
        with torch.no_grad():
            def _text_length(text):
                if isinstance(text, dict):              #{key: value} case
                    return len(next(iter(text.values())))
                elif not hasattr(text, '__len__'):      #Object has no len() method
                    return 1
                elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
                    return len(text)
                else:
                    return sum([len(t) for t in text])      #Sum of length of individual strings

            all_embeddings = []
            length_sorted_idx = np.argsort([_text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in range(0, len(sentences), batch_size):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                embeddings = self.encode_text(sentences_batch, projection=False).cpu()
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings = torch.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings
    
    def encode_text(self, texts, projection=False):

        encoded_input = self.tokenizer(texts, padding=True, truncation=True,return_tensors="pt")
        encoded_input = {
            'input_ids': encoded_input['input_ids'].to(self.device),
            'attention_mask': encoded_input['attention_mask'].to(self.device)
            }
        text_features = self.text_backbone(**encoded_input)
        text_features = mean_pooling(text_features, encoded_input['attention_mask'])

        return text_features


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__=='__main__':
    
    
    huggingface_transformers = [
        'roberta-large',
        'roberta-large-mnli',
        'facebook/muppet-roberta-large',
        'Jean-Baptiste/roberta-large-ner-english',
        'princeton-nlp/unsup-simcse-roberta-large',
        'princeton-nlp/sup-simcse-roberta-large',
        'sentence-transformers/all-roberta-large-v1',
        'xlm-roberta-large',
        'xlm-roberta-large-finetuned-conll03-english',
        'deepset/xlm-roberta-large-squad2',
        'joeddav/xlm-roberta-large-xnli'
    ]

    sentence_transformers = [
        'average_word_embeddings_glove.6B.300d',
        'clip-ViT-B-32',
        'clip-ViT-B-16',
        'clip-ViT-L-14',
    ]


for model_name in sentence_transformers + huggingface_transformers:
    if model_name in huggingface_transformers:
        model = WrappedHuggingfaceTransformer(model_name).cuda()
    elif model_name in sentence_transformers:
        model = SentenceTransformer(model_name).cuda()
    print('= '*64)
    print(model_name)

    for dataset in ['coco_train']:
        if dataset=='captioned_imagenet_val':
            captions, labels = get_captioned_imagenet(split='val')
        if dataset=='captioned_imagenet_train':
            captions, labels = get_captioned_imagenet(split='train')
        elif dataset=='coco_val':
            captions, labels = get_coco(split='val')
        elif dataset=='coco_train':
            captions, labels = get_coco(split='train')
        elif dataset=='imagenet_captions':
            captions, labels = get_imagenet_captions()

        features = model.encode(captions, convert_to_tensor=True, show_progress_bar=True, batch_size=1024)
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
                
        #save_tsne(features, labels, dataset + '-' + model_name + '.png')
        
        print(dataset, 'TransRate', transrate(features, labels, eps=1e-4))
        print(dataset, 'logme', log_maximum_evidence(features, labels))
        print(dataset, 'h_score', h_score(features, labels))

