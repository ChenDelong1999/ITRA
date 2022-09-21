
import torch
import torch.nn as nn
import os
import tqdm
import pandas as pd
import numpy as np
from torch import nn
from training.evaluations.coco_retrieval import CocoCaptions


from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from mutual_information.transrate import transrate
from mutual_information.logme import log_maximum_evidence
from mutual_information.hscore import h_score
from mutual_information.mean_classifier import mean_classifier_acc

from openTSNE import TSNE
from utils.captioned_imagenet import CaptionedImageNet
import matplotlib.pyplot as plt

import open_clip

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
    for i in range(len(all_captions)):
        all_captions[i] = all_captions[i][:50]

    return all_captions, all_labels


def get_captioned_imagenet(split='val'):
    dataset = CaptionedImageNet(path='data/captioned_imagenet', split=split, preprocess=None)
    all_labels = np.array(dataset.labels)
    all_captions = dataset.captions

    return all_captions, all_labels


def get_built_knn(csv):
    df = pd.read_csv(csv, sep='\t')
    labels = np.array(df['labels'].tolist())
    texts = np.array(df['texts'].tolist())

    return texts, labels

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
    def __init__(self, model_name) -> None:
        super().__init__()        
        self.model_name = model_name
        # config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_backbone = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
                
        if self.model_name == 'facebook/contriever-msmarco':    
            text_features = mean_pooling_contriever(text_features[0], encoded_input['attention_mask'])
        else:
            text_features = mean_pooling(text_features, encoded_input['attention_mask'])

        return text_features


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_contriever(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class WrappedOpenCLIPTransformer(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained='openai',
            precision='amp',
            device='cuda',
            args=None
        )
        self.device = 'cuda'
        self.text_backbone = CLIP_model
        self.tokenizer = open_clip.tokenize
        
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

            iterator = tqdm.tqdm(range(0, len(sentences), batch_size)) if show_progress_bar else range(0, len(sentences), batch_size)
            for start_index in iterator:
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                embeddings = self.encode_text(sentences_batch, projection=True).cpu()
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings = torch.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings
    
    def encode_text(self, texts, projection=False):
        with torch.no_grad():
            texts = self.tokenizer(texts, context_length=77).to(self.device)
            def open_clip_forward(texts):
                x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model] (64, 77-args.n_prompts, 512)
                x = x + self.text_backbone.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.text_backbone.ln_final(x) # [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                return x
            text_features = open_clip_forward(texts)
        return text_features

if __name__=='__main__':
    

    sentence_transformers = [
        'average_word_embeddings_glove.6B.300d',
        'average_word_embeddings_komninos',
    ]

    open_clip_transformers = [
        # 'RN50',
        # 'RN101',
        # 'RN50x4',
        # 'RN50x16',
        'ViT-B-32',
        'ViT-B-16',
        'ViT-L-14',
        'RN50x64',
        'ViT-L-14-336'
    ]

    huggingface_transformers = [
        # 'bert-base-uncased',
        # 'bert-base-cased',
        # 'bert-large-uncased',
        # 'bert-large-cased',
        'roberta-base',
        'roberta-large',
        'roberta-large-mnli',
        'facebook/muppet-roberta-large',
        'facebook/contriever',
        'facebook/contriever-msmarco',
        'Jean-Baptiste/roberta-large-ner-english',
        'princeton-nlp/unsup-simcse-roberta-large',
        'princeton-nlp/sup-simcse-roberta-large',
        'xlm-roberta-large',
        'xlm-roberta-large-finetuned-conll03-english',
        'deepset/xlm-roberta-large-squad2',
        'joeddav/xlm-roberta-large-xnli',
        'sentence-transformers/distiluse-base-multilingual-cased-v2',
        'sentence-transformers/paraphrase-distilroberta-base-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        'sentence-transformers/msmarco-distilbert-base-tas-b',
        'sentence-transformers/all-MiniLM-L12-v1',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/all-roberta-large-v1',
    ]

    built_csv='k=10_n_query=10000_feature=resnet18-yfcc14m.npy.csv'
    print(built_csv)

    #for model_name in sentence_transformers + huggingface_transformers + open_clip_transformers :
    for model_name in open_clip_transformers:        
        if model_name in huggingface_transformers:
            model = WrappedHuggingfaceTransformer(model_name).cuda()
        elif model_name in sentence_transformers:
            model = SentenceTransformer(model_name).cuda()
        elif model_name in open_clip_transformers:
            model = WrappedOpenCLIPTransformer(model_name).cuda()
        print('= '*64)
        print(model_name)

        # All avaliable datasets
        # for dataset in ['captioned_imagenet_train', 'captioned_imagenet_val', 'coco_train', 'coco_val', 'imagenet_captions']:
        for dataset in ['captioned_imagenet_val', 'coco_val', 'imagenet_captions']:
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
            elif dataset=='built_knn':
                captions, labels = get_built_knn(built_csv)

            features = model.encode(captions, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
            features = torch.nn.functional.normalize(features, dim=-1)
            features = features.cpu().numpy()
                    
            # save_tsne(features, labels, dataset + '-' + model_name + '.png')     
            # print(dataset, 'TransRate', transrate(features, labels, eps=1e-4))
            # print(dataset, 'logme', log_maximum_evidence(features, labels))
            # print(dataset, 'h_score', h_score(features, labels))
            print(dataset, '[mean_classifier_acc]\t', mean_classifier_acc(features, labels))

