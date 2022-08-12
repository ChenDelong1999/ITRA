import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

from training.evaluations.nlp_evaluations import nlp_eval
from training.params import parse_args

from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

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


""" Teacher Zoo
    --text-model 'RN50' --text-model-builder 'OpenCLIP' --pretrained-text-model --text-head-n-layers 0 \ CLIP-pretrained
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 
        --text-model 'roberta-large' \                              # https://huggingface.co/roberta-large
        --text-model 'roberta-large-mnli' \                         # https://huggingface.co/roberta-large-mnli
        --text-model 'facebook/muppet-roberta-large'                # https://huggingface.co/facebook/muppet-roberta-large
        --text-model 'Jean-Baptiste/roberta-large-ner-english' \    # https://huggingface.co/Jean-Baptiste/roberta-large-ner-english
        --text-model 'princeton-nlp/unsup-simcse-roberta-large'     # https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large
        --text-model 'princeton-nlp/sup-simcse-roberta-large'       # https://huggingface.co/princeton-nlp/sup-simcse-roberta-large
        --text-model 'sentence-transformers/all-roberta-large-v1' \ # https://huggingface.co/sentence-transformers/all-roberta-large-v1
        --text-model 'xlm-roberta-large' \                          # https://huggingface.co/xlm-roberta-large
        --text-model 'xlm-roberta-large-finetuned-conll03-english'\ # https://huggingface.co/xlm-roberta-large-finetuned-conll03-english
        --text-model 'deepset/xlm-roberta-large-squad2' \           # https://huggingface.co/deepset/xlm-roberta-large-squad2
        --text-model 'joeddav/xlm-roberta-large-xnli' \             # https://huggingface.co/joeddav/xlm-roberta-large-xnli
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 
        --text-model 'all-mpnet-base-v2' \
        --text-model 'average_word_embeddings_glove.6B.300d' \
"""

if __name__=='__main__':

    args = parse_args()
    args.nlp_eval_frequency = 1
    args.distributed = False
    args.eval_data_dir='/data/Datasets'
    args.fast_evaluation=False
    models = [
        'roberta-large',
        'roberta-large-mnli',
        'facebook/muppet-roberta-large',
        'Jean-Baptiste/roberta-large-ner-english',
        'princeton-nlp/unsup-simcse-roberta-large',
        'sentence-transformers/all-roberta-large-v1',
        'xlm-roberta-large',
        'xlm-roberta-large-finetuned-conll03-english',
        'deepset/xlm-roberta-large-squad2',
        'joeddav/xlm-roberta-large-xnli'
    ]

    # models = [
    #     'average_word_embeddings_glove.6B.300d',
    #     'clip-ViT-B-32',
    #     'clip-ViT-B-16',
    #     'clip-ViT-L-14',
    #     'clip-ViT-B-32-multilingual-v1',
    # ]

    df = None
    for model_name in models:
        model = WrappedHuggingfaceTransformer(model_name).cuda()
        #model = SentenceTransformer(model_name).cuda()
        results = nlp_eval(model, epoch=0, args=args)
        results['model'] = model_name
        logging.info( f"Finished evaluation of {model_name}\n" + "\n".join([f"\t{k}\t{v}" for k, v in results.items()]))

        line = pd.DataFrame(results, index = [0])
        if df is None:
            df = line
        else:
            df = pd.concat([df, line])
        print(df)
        
    df.to_csv('eval_all_RoBERTa-large.csv', sep='\t', index=None)
