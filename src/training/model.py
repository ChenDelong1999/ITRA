import logging
import random
import os
import torch
import torch.nn as nn
import numpy as np

import torchvision
import open_clip
import cn_clip.clip as cn_clip
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForSequenceClassification

import transformers.adapters
from sentence_transformers import SentenceTransformer

from training.distributed import is_master
from training.projection import DINOHead
from training.prompt import Prompt
import training.transforms

from distiller import NEED_LOGIT_SCALE, NEED_PROTOTYPE_LAYER
from contextlib import suppress


def get_model(args):
    logging.info(f'Builing model for rank {args.rank}')
    
    # === text model === #
    if is_master(args):
        logging.info(f'Loading [{args.text_model}] as text model via [{args.text_model_builder}]. Pretrained={args.pretrained_text_model}')
    
    if args.text_model_builder=='openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.text_model,
            pretrained=args.text_model_tag if args.pretrained_text_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        CLIP_model.visual = None
        text_backbone = CLIP_model
        tokenizer = open_clip.tokenize
        args.text_width, args.text_dim = text_backbone.text_projection.size()
                    
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
            
    elif args.text_model_builder=='chineseclip':
        CLIP_model, preprocess_val = cn_clip.load_from_name(
            name=args.text_model, 
            device=args.device, 
            download_root=os.path.join(args.cache_dir, 'cn_clip')
            )
        preprocess_train = preprocess_val # TODO: add data augmentations
        
        CLIP_model.visual = None
        text_backbone = CLIP_model
        tokenizer = cn_clip.tokenize
        args.text_width, args.text_dim = text_backbone.text_projection.size()
                    
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
        
    elif args.text_model_builder=='huggingface':
        if not args.pretrained_text_model and is_master(args):
            logging.info(f'huggingface-transormer uses pretrained weight by default!')
        config = AutoConfig.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        
        if 'Taiyi' in args.text_model:
            # Taiyi models need classification heads, using AutoModel will skip loading them.
            # See https://huggingface.co/IDEA-CCNL/Taiyi-Roberta-124M-D-v2
            text_backbone = BertForSequenceClassification.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        else:
            text_backbone = AutoModel.from_pretrained(args.text_model, cache_dir=os.path.join(args.cache_dir, 'huggingface'))
        args.text_dim = config.hidden_size
        args.text_width = None
                
        if args.adapter is not None:
            if args.adapter=='bottleneck_adapter':
                config = transformers.adapters.AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            elif args.adapter=='prefix_tuning':
                config = transformers.adapters.PrefixTuningConfig(flat=False, prefix_length=30)
            elif args.adapter=='lang_adapter':
                config = transformers.adapters.PfeifferInvConfig()
            elif args.adapter=='lora_adapter':
                config = transformers.adapters.LoRAConfig(r=8, alpha=16)
            elif args.adapter=='dummy':
                config = transformers.adapters.CompacterConfig()
            elif args.adapter=='ia3_adapter':
                config = transformers.adapters.IA3Config()
            elif args.adapter=='mam_adapter':
                config = transformers.adapters.MAMConfig()
            elif args.adapter=='unipelt':
                config = transformers.adapters.UniPELTConfig()
            else:
                raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
            
            logging.info(f'[Adapter]: Using adapter: {args.adapter}.')
            text_backbone.add_adapter(args.adapter, config=config)
            text_backbone.train_adapter(args.adapter)
            
    elif args.text_model_builder=='sbert':
        if not args.pretrained_text_model and is_master(args):
            logging.info(f'Sentence-transormer uses pretrained weight by default!')
        text_backbone = SentenceTransformer(args.text_model, device=args.device).to(args.device)
        tokenizer = None
        args.text_dim = text_backbone.get_sentence_embedding_dimension()
        args.text_width = args.text_dim
        
        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
    
    else:
        raise RuntimeError(f'text model builder "{args.text_model_builder}" is not supported.')
    
    
    # === image model === #
    if is_master(args):
        logging.info(f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')
    
    if args.image_model_builder=='openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.image_model,
            pretrained=args.image_model_tag if args.pretrained_image_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim

        
    elif args.image_model_builder=='chineseclip':
        CLIP_model, preprocess_val = cn_clip.load_from_name(
            name=args.text_model, 
            device=args.device, 
            download_root=os.path.join(args.cache_dir, 'cn_clip')
            )

        preprocess_train = preprocess_val # TODO: add data augmentations        
        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim
    
    elif args.image_model_builder=='torchvision':
        os.environ['TORCH_HOME'] = os.path.join(args.cache_dir, 'torchvision')
        image_backbone = torchvision.models.__dict__[args.image_model](pretrained=args.pretrained_image_model, num_classes=1000)

        LAST_FC = ['resnet', 'shufflenet', 'convnext', 'regnet', 'inception']
        CLASSIFIER_WITH_DROPOUT = ['alexnet', 'squeezenet', 'mnasnet', 'efficientnet']
        CLASSIFIER_WITHOUT_DROPOUT = ['mobilenet', 'vgg', 'densenet', 'googlenet']
        LAST_HEAD = ['vit']

        if 'resnet' in args.image_model or 'shufflenet' in args.image_model or 'convnext' in args.image_model or 'regnet' in args.image_model or 'inception' in args.image_model:
            image_backbone.output_dim = image_backbone.fc.weight.shape[1]
            image_backbone.fc=torch.nn.Identity()
        if 'alexnet' in args.image_model or 'squeezenet' in args.image_model or 'mnasnet' in args.image_model or 'efficientnet' in args.image_model: 
            # there are dropout layer between first linear layer in the classifier # TODO: squeeze net has conv2d instead of linear
            image_backbone.output_dim = image_backbone.classifier[1].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        if 'mobilenet' in args.image_model or 'vgg' in args.image_model or 'densenet' in args.image_model or 'googlenet' in args.image_model :
            image_backbone.output_dim = image_backbone.classifier[0].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        if 'vit' in args.image_model:
            image_backbone.output_dim = image_backbone.hidden_dim
            image_backbone.heads=torch.nn.Identity()
            image_backbone.head=torch.nn.Identity()
        image_backbone.to(device=args.device)

        preprocess_train, preprocess_val = training.transforms.preprocess_train, training.transforms.preprocess_val
    
    
    elif args.image_model_builder=='torchhub':
        if not args.pretrained_image_model and is_master(args):
            logging.info(f'Torch hub uses pretrained weight by default!')

        torch.hub.set_dir(os.path.join(args.cache_dir, 'torchhub'))
        
        # https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True 

        image_backbone = torch.hub.load(args.image_model_tag, args.image_model)
        if 'resnet' in args.image_model:
            if 'vicreg' in args.image_model_tag:
                image_backbone.output_dim = 2048 # TODO: check ResNet-50 (x2) and ResNet-200 (x2)
            
        if 'vit' in args.image_model or 'xcit' in args.image_model:            
            image_backbone.output_dim = image_backbone.norm.weight.size(0)

        if 'regnety' in args.image_model:            
            image_backbone.output_dim = 2048

        image_backbone.to(device=args.device)
        preprocess_train, preprocess_val = training.transforms.preprocess_train, training.transforms.preprocess_val

    else:
        raise RuntimeError(f'image model builder "{args.image_model_builder}" is not supported.')
   
    # Set required_grad
    for name, param in text_backbone.named_parameters():
        param.requires_grad = False if args.lock_text_model else True

    for name, param in text_backbone.named_parameters():
        param.requires_grad = False if args.lock_image_model else True

    model = WrappedModel(
        text_backbone=text_backbone, 
        image_backbone=image_backbone, 
        tokenizer=tokenizer, 
        args=args
        )
        
    return model, preprocess_train, preprocess_val, preprocess_val



class WrappedModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args, prompt=None) -> None:
        super().__init__()
        self.device = args.device
        self.text_model = args.text_model
    
    # text backbone
        self.text_backbone = text_backbone
        self.text_pooler = args.text_pooler
        if self.text_pooler!= 'cls':
            self.text_backbone.pooler = nn.Identity()
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer = tokenizer        
        self.text_model_builder = args.text_model_builder
        self.image_model_builder = args.image_model_builder
        self.max_seq_length = args.max_seq_length
            
        self.image_context = torch.no_grad if args.lock_image_model else suppress 
        self.text_context = torch.no_grad if (args.lock_text_model and args.adapter is None) else suppress
        
        if is_master(args):
            logging.info(f'Calculate gradients for image backbone?\t{self.image_context==suppress}')
            logging.info(f'Calculate gradients for text backbone?\t{self.text_context==suppress}')
            logging.info(f'image_context: {str(self.image_context)}')
            logging.info(f'text_context: {str(self.text_context)}')
        
        # TODO: CoOp text prompt (optional) 
        if args.prompt:
            self.prompt = nn.Parameter(torch.empty(args.n_prompt, args.text_width))
            torch.nn.init.normal_(self.prompt, std=0.02)
            self.n_prompt = args.n_prompt
        else:
            self.prompt = None

    # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim
        self.image_model_tag = args.image_model_tag

    
    # text projection head
        if args.text_head_n_layers > 0 or args.distiller in NEED_PROTOTYPE_LAYER:
            if args.image_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.image_dim # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=args.distiller not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            
            # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.distiller in NEED_PROTOTYPE_LAYER and args.teacher=='text':
                for param in self.text_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.text_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Text backbone do not append projection head, so set args.joint_projection_dim = self.text_dim')
            args.joint_projection_dim = self.text_dim

    # image projection head
        if args.image_head_n_layers > 0 or args.distiller in NEED_PROTOTYPE_LAYER:
            if args.text_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.text_dim # adaption layer
            self.image_projection_head = DINOHead(
                in_dim=self.image_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.image_head_n_layers, skip_last_layer=args.distiller not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            # FIXME? # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.distiller in NEED_PROTOTYPE_LAYER and args.teacher=='image':
                for param in self.image_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        if args.distiller in NEED_LOGIT_SCALE:
            if hasattr(self.text_backbone, 'logit_scale'):
                self.logit_scale = self.text_backbone.logit_scale 
                self.text_backbone.logit_scale = None
            else:
                self.logit_scale = torch.autograd.Variable(torch.ones(1) * np.log(1 / args.logit_scale)).to(self.device)
            self.logit_scale = nn.Parameter(self.logit_scale)
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale = torch.zeros(1)
        self.to(self.device)

    def reinit_logit_scale(self, logit_scale):
        self.logit_scale = torch.nn.parameter.Parameter(torch.ones(1) * np.log(1 / logit_scale))#.to(self.device)
        #self.logit_scale.to(self.device)
        self.to(self.device)

    def encode_image(self, images, projection=False):
        with self.image_context():
            if self.image_model_builder=='chineseclip': 
                images = images.type(self.image_backbone.conv1.weight.dtype)
            image_features = self.image_backbone(images)
            if 'vicregl' in self.image_model_tag:
                image_features = image_features[1]
        if projection:
            image_features = self.image_projection_head(image_features)
        return image_features.float()

    # sentence-transformers API
    def encode(self, sentences, batch_size=32, show_progress_bar=None, convert_to_numpy=True, convert_to_tensor=True, use_pooler=False):
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
                embeddings = self.encode_text(sentences_batch, projection=True, use_pooler=use_pooler).cpu()
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings = torch.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings
    
    def encode_text(self, texts, projection=False, use_pooler=True):
        with self.text_context():
            if self.text_model_builder in ['openclip']:
                # TODO: support CoOp-style prompting (CoOp for retrieval finetuning?)
                context_length = (77 - self.n_prompt) if self.prompt is not None else 77
                texts = self.tokenizer(texts, context_length=context_length).to(self.device)
                def open_clip_forward(texts):
                    x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model] (bs, 77-args.n_prompts, 512)
                    if self.prompt is not None:
                        batch_prompt = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
                        x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
                    x = x + self.text_backbone.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.text_backbone.ln_final(x) # [batch_size, n_ctx, transformer.width]
                    # take features from the eot embedding (eot_token is the highest number in each sequence)
                    x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                    return x
                text_features = open_clip_forward(texts)
            
            elif self.text_model_builder=='chineseclip': 
                # text_features = self.text_backbone.encode_text(self.tokenizer(texts).to(self.device))
                # Re-write encode_text() function to avoid reading missing .dtype()
                # For https://github.com/OFA-Sys/Chinese-CLIP/blob/ce534cc8c0dde1206cd3e3ddf7e4023455c83450/cn_clip/clip/model.py#L375
                
                def chineseclip_encode_text(text):
                    pad_index = self.text_backbone.tokenizer.vocab['[PAD]']
                    attn_mask = text.ne(pad_index).type(self.text_backbone.text_projection.dtype)
                    x = self.text_backbone.bert(text, attention_mask=attn_mask)[0].type(self.text_backbone.text_projection.dtype) # [batch_size, seq_length, hidden_size]
                    return x[:, 0, :] @ self.text_backbone.text_projection

                text_features = chineseclip_encode_text(self.tokenizer(texts).to(self.device)).float()

            elif self.text_model_builder=='sbert':            
                texts = self.text_backbone.tokenize(texts)
                texts = {
                    'input_ids': texts['input_ids'].to(self.device),
                    'attention_mask': texts['attention_mask'].to(self.device)
                    }
                text_features = self.text_backbone(texts)
                sentence_embedding = text_features['sentence_embedding']
                text_features = sentence_embedding
                #token_embeddings = text_features['token_embeddings']
                #text_features = token_embeddings[:, 0, :].contiguous()

            elif self.text_model_builder=='huggingface':           
                # Preprocess
                if self.text_pooler == 'PromptBERT':
                    texts_lengths = [] # memorize the number of token of each sentence for position id padding
                    for t in range(len(texts)):
                        encoded_sentence = self.tokenizer.encode(texts[t], truncation=True, max_length=self.max_seq_length)
                        texts_lengths.append(len(encoded_sentence))
                        sentence = self.tokenizer.decode(encoded_sentence, skip_special_tokens=True)

                        if self.text_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased']:
                            if random.random() > 0.5 or not self.training:
                                texts[t] = f'The sentence of "{sentence}" means {self.tokenizer.mask_token}.'
                            else:
                                texts[t] = f'This sentence : "{sentence}" means {self.tokenizer.mask_token}.'
                        else: # roberta
                            if random.random() > 0.5 or not self.training:
                                texts[t] = f"This sentence : '{sentence}' means {self.tokenizer.mask_token}."
                            else:
                                texts[t] = f"The sentence : '{sentence}' means {self.tokenizer.mask_token}."

                    texts_lengths = np.array(texts_lengths)
                    encoded_input = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
                else:
                    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_length)
                
                # To GPU
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(self.device),
                    'attention_mask': encoded_input['attention_mask'].to(self.device)
                    }

                # Forward
                outputs = self.text_backbone(**encoded_input, output_hidden_states=True, return_dict=True)
                    # last_hidden = outputs.last_hidden_state   # (batch_size, sequence_length, hidden_size)
                    # pooler_output = outputs.pooler_output     # (batch_size, hidden_size)
                    # hidden_states = outputs.hidden_states     # (batch_size, sequence_length, hidden_size) x layers (tuple)

                # Pooling
                # use_pooler = True if self.training else False # TODO: there is a conflict between retrieval evaluation and sts evaluation...
                if self.text_pooler=='mean':
                    text_features = mean_pooling(outputs.last_hidden_state, encoded_input['attention_mask'])

                elif self.text_pooler=='cls' and use_pooler:
                    text_features = outputs.pooler_output
                
                elif (self.text_pooler=='cls' and not use_pooler) or (self.text_pooler == 'cls_before_pooler'):
                    text_features = outputs.last_hidden_state[:, 0].contiguous()

                elif self.text_pooler == 'PromptBERT':
                    # Retrieve [mask] token
                    text_features = outputs.last_hidden_state[encoded_input['input_ids'] == self.tokenizer.mask_token_id].contiguous()
                    # Template Denoising
                    with torch.no_grad():
                        if self.text_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased']:
                            encoded_input_delta = self.tokenizer(f'The sentence of " " means {self.tokenizer.mask_token}.', return_tensors="pt")
                        else:
                            encoded_input_delta = self.tokenizer(f"This sentence : ' ' means {self.tokenizer.mask_token}.", return_tensors="pt")
                            
                        encoded_input_delta = {
                            'input_ids': encoded_input_delta['input_ids'].repeat(len(texts), 1).to(self.device),
                            'attention_mask': encoded_input_delta['attention_mask'].repeat(len(texts), 1).to(self.device),
                        }
                        delta_position_ids = torch.arange(len(encoded_input_delta['input_ids'][0])).long().repeat(len(texts), 1)
                        # (0) <Start> | (1) This | (2) sentence | (3) of/: | (4) "/' | (5) {sentence} ...
                        delta_position_ids[:,5:] += texts_lengths.reshape(len(texts), 1) 
                        delta = self.text_backbone(**encoded_input_delta, position_ids=delta_position_ids.to(self.device), output_hidden_states=True, return_dict=True)
                        delta = delta.last_hidden_state[encoded_input_delta['input_ids'] == self.tokenizer.mask_token_id]
                    text_features -= delta
                
                elif self.text_pooler == 'logits':
                    text_features = outputs.logits
                    

        if projection:
            text_features = self.text_projection_head(text_features)

        return text_features
    
    def forward(self, images, texts, text_only):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        texts:  torch.tensor (batchs_size, token_indexs)
        """
        text_features = self.encode_text(texts, projection=True)

        if text_only: # skip image forward for efficient teacher caching 
            image_features = text_features
        else:
            image_features = self.encode_image(images, projection=True)

        return image_features, text_features, self.logit_scale.exp()
        
def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
