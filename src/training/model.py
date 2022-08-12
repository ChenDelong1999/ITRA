import logging
import torch
import torch.nn as nn
import numpy as np
import open_clip
#from open_clip import trace_model, create_model_and_transforms, create_transforms, list_models, tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModel
#from seed import models
import torchvision
from training.distributed import is_master
from training.projection import DINOHead
from training.prompt import Prompt

from transformers.adapters import PrefixTuningConfig
from transformers.adapters import AdapterConfig
from transformers.adapters import PfeifferInvConfig
#from transformers.adapters import LoRAConfig
from transformers.adapters import CompacterConfig            
from transformers.adapters import MAMConfig

from distiller import NEED_LOGIT_SCALE, NEED_PROTOTYPE_LAYER

from contextlib import suppress
def get_model(args):
    # === text model === #
    if is_master(args):
        logging.info(f'Loading [{args.text_model}] as text model via [{args.text_model_builder}]. Pretrained={args.pretrained_text_model}')
    
    if args.text_model_builder=='OpenCLIP':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.text_model,
            pretrained='openai' if args.pretrained_text_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            args=args
        )
        CLIP_model.visual = None
        text_backbone = CLIP_model
        tokenizer = open_clip.tokenize
        args.text_dim = text_backbone.embed_dim
        args.text_width = text_backbone.text_width
    
    elif args.text_model_builder=='sentence-transformer':
        if not args.pretrained_text_model and is_master(args):
            logging.info(f'Sentence-transormer uses pretrained weight by default!')
        text_backbone = SentenceTransformer(args.text_model, device=args.device).to(args.device)
        tokenizer = None
        args.text_dim = text_backbone.get_sentence_embedding_dimension()
        args.text_width = args.text_dim
    
    elif args.text_model_builder=='huggingface-transformer':
        config = AutoConfig.from_pretrained(args.text_model)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        text_backbone = AutoModel.from_pretrained(args.text_model)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        args.text_dim = config.hidden_size
        args.text_width = None

        if args.adapter=='prefix_tuning':
            config = PrefixTuningConfig(flat=False, prefix_length=args.n_prompt)
            text_backbone.add_adapter("prefix_tuning", config=config)
            text_backbone.train_adapter("prefix_tuning")
            logging.info(f'[Adapter]: prefix_tuning adapter have been added!')

        elif args.adapter=='bottleneck_adapter':
            config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            text_backbone.add_adapter("bottleneck_adapter", config=config)
            text_backbone.train_adapter("bottleneck_adapter")
            logging.info(f'[Adapter]: bottleneck_adapter adapter have been added!')

        elif args.adapter=='lang_adapter':
            config = PfeifferInvConfig()
            text_backbone.add_adapter("lang_adapter", config=config)
            text_backbone.train_adapter("lang_adapter")
            logging.info(f'[Adapter]: lang_adapter adapter have been added!')
        
        # elif args.adapter=='lora_adapter':
        #     config = LoRAConfig(r=8, alpha=16)
        #     text_backbone.add_adapter("lora_adapter", config=config)
        #     text_backbone.train_adapter("lang_adapter")
        #     logging.info(f'[Adapter]: lora_adapter adapter have been added!')
    
        elif args.adapter=='dummy':
            config = CompacterConfig()
            text_backbone.add_adapter("dummy", config=config)
            text_backbone.train_adapter("dummy")
            logging.info(f'[Adapter]: dummy adapter have been added!')

        elif args.adapter=='mam_adapter':
            config = MAMConfig()
            text_backbone.add_adapter("mam_adapter", config=config)
            text_backbone.train_adapter("mam_adapter")
            logging.info(f'[Adapter]: mam_adapter adapter have been added!')
        # else:
        #     text_backbone.freeze_model(False)



    # === image model === #
    if is_master(args):
        logging.info(f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')
    
    if args.image_model_builder=='OpenCLIP':
        CLIP_model, _, _ = open_clip.create_model_and_transforms(
            model_name=args.image_model,
            pretrained='openai' if args.pretrained_image_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            args=args
        )
        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim
    
    elif args.image_model_builder=='torchvision':
        image_backbone = torchvision.models.__dict__[args.image_model](pretrained=args.pretrained_image_model, num_classes=1000)
        if 'resnet' in args.image_model or 'shufflenet' in args.image_model or 'convnext' in args.image_model:
            image_backbone.output_dim = image_backbone.fc.weight.shape[1]
            image_backbone.fc=torch.nn.Identity()
        if 'alexnet' in args.image_model:
            image_backbone.output_dim = image_backbone.classifier[1].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        if 'mobilenet' in args.image_model:
            image_backbone.output_dim = image_backbone.classifier[0].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        image_backbone.to(device=args.device)
        
    for param in image_backbone.parameters():
        param.requires_grad = True if args.unlock_image_model else False
    
    if args.adapter is None:
        for name, param in text_backbone.named_parameters():
            param.requires_grad = True if args.unlock_text_model or 'prefix_tuning' in name else False
    
    model = WrappedModel(
        text_backbone=text_backbone, 
        image_backbone=image_backbone, 
        tokenizer=tokenizer, 
        args=args
        )
    
    preprocess_train, preprocess_val, preprocess_aug = open_clip.create_transforms(image_size=224, args=args)
            
    if is_master(args):
        logging.info('Model created\n' +str(model))
    
    return model, preprocess_train, preprocess_val, preprocess_aug



class WrappedModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args, prompt=None) -> None:
        super().__init__()
        self.device = args.device
        
        # text backbone
        self.text_backbone = text_backbone
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer = tokenizer        
        self.text_model_builder = args.text_model_builder
        self.image_context = suppress if args.unlock_image_model else torch.no_grad
        self.text_context = suppress if (args.unlock_text_model or args.prompt or args.adapter is not None) else torch.no_grad
        if is_master(args):
            logging.info(f'image_context: {str(self.image_context)}')
            logging.info(f'text_context: {str(self.text_context)}')
        self.unlock_text_model = args.unlock_text_model
        
        # text prompt (optional)
        if args.prompt:
            self.prompt = nn.Parameter(torch.empty(args.n_prompt, args.text_width))
            torch.nn.init.normal_(self.prompt, std=0.02)
            self.n_prompt = args.n_prompt
        else:
            self.prompt = None

        # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim

        if self.text_dim!=self.image_dim and args.text_head_n_layers+args.image_head_n_layers==0:
            raise AssertionError(f'text and backbone feature dimension do not match ({self.text_dim} vs {self.image_dim}), projection head nlayer > 0 is needed!')

        # text projection head
        if args.text_head_n_layers > 0 or args.distiller in NEED_PROTOTYPE_LAYER:
            if args.image_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.image_dim # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=args.distiller not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            
            # ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.distiller in NEED_PROTOTYPE_LAYER:
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
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        if args.distiller in NEED_LOGIT_SCALE:
            self.logit_scale = self.text_backbone.logit_scale if hasattr(self.text_backbone, 'logit_scale') else torch.autograd.Variable(torch.ones(1) * np.log(1 / 0.07)).to(self.device)
            self.logit_scale = nn.Parameter(self.logit_scale)
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale = torch.zeros(1)
        self.to(self.device)


    def encode_image(self, images, projection=False):
        with self.image_context():
            image_features = self.image_backbone(images)
        if projection:
            image_features = self.image_projection_head(image_features)
        return image_features

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
        with self.text_context():
            if self.text_model_builder=='OpenCLIP':
                context_length = (77 - self.n_prompt) if self.prompt is not None else 77
                texts = self.tokenizer(texts, context_length=context_length).to(self.device)
                def open_clip_forward(texts):
                    x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model] (64, 77-args.n_prompts, 512)
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
            
            elif self.text_model_builder=='sentence-transformer':            
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

            elif self.text_model_builder=='huggingface-transformer':            
                encoded_input = self.tokenizer(texts, padding=True, truncation=True,return_tensors="pt")
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(self.device),
                    'attention_mask': encoded_input['attention_mask'].to(self.device)
                    }
                text_features = self.text_backbone(**encoded_input)
                text_features = mean_pooling(text_features, encoded_input['attention_mask'])

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
        
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

