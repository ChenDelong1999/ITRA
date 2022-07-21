import logging
import torch
import torch.nn as nn
import numpy as np
import open_clip
#from open_clip import trace_model, create_model_and_transforms, create_transforms, list_models, tokenize
from sentence_transformers import SentenceTransformer
import transformers
from transformers import AutoTokenizer
#from seed import models
import torchvision
from training.distributed import is_master
from training.projection import DINOHead

def get_model(args):

    # loading via transformer
    #     #text_teacher = SentenceTransformer(args.text_teacher, device=args.device).to(args.device)
    #     from transformers import AutoTokenizer, AutoModel
    #     tokenizer = AutoTokenizer.from_pretrained(args.text_teacher)
    #     text_teacher = AutoModel.from_pretrained(args.text_teacher).to(args.device)
    #     text_teacher.max_seq_length = 77
    #     if args.text_teacher in ['clip-ViT-B-32', 'clip-ViT-B-16']:
    #         args.text_teacher_dim = 512
    #     elif args.text_teacher in ['microsoft/mpnet-base', 'roberta-base']:   
    #         args.text_teacher_dim = 768
    #     distiller_text = get_distiller(args.distiller)(args, args.text_teacher_dim).to(args.device)


    # if args.prompt:
    #     prompt = Prompt(args.n_prompt, args.text_embedding_dim, args)
    #     text_teacher.prompt = prompt
    # else:
    #     prompt = None
    #     text_teacher.prompt = None

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
        args.text_width = None
        # #text_backbone.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(args.device)
        # text_backbone.logit_scale = torch.autograd.Variable(torch.ones([]) * np.log(1 / 0.07)).to(args.device)


    # === image model === #
    if is_master(args):
        logging.info(f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')
    if args.image_model_builder=='OpenCLIP':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
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
        if 'resnet' in args.image_model:
            image_backbone.output_dim = image_backbone.fc.weight.shape[1]
            image_backbone.fc=torch.nn.Identity()
        if 'alexnet' in args.image_model:
            image_backbone.output_dim = image_backbone.classifier[1].weight.shape[1]
            image_backbone.classifier=torch.nn.Identity()
        image_backbone.to(device=args.device)
        
    for param in image_backbone.parameters():
        param.requires_grad = True if args.unlock_image_model else False
    for param in text_backbone.parameters():
        param.requires_grad = True if args.unlock_text_model else False
    
    model = WrappedModel(text_backbone=text_backbone, image_backbone=image_backbone, tokenizer=tokenizer, args=args)
    
    preprocess_train, preprocess_val = open_clip.create_transforms(image_size=224, args=args)

            
    if is_master(args):
        logging.info('model created\n' +str(model))
    
    return model, preprocess_train, preprocess_val



class WrappedModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args) -> None:
        super().__init__()
        self.device = args.device
        
        # text backbone
        self.text_backbone = text_backbone
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer = tokenizer        
        self.text_model_builder = args.text_model_builder

        # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim

        if self.text_dim!=self.image_dim and args.text_head_n_layers+args.image_head_n_layers==0:
            raise AssertionError(f'text and backbone feature dimension do not match ({self.text_dim} vs {self.image_dim}), projection head nlayer > 0 is needed!')

        # text projection head
        if args.text_head_n_layers>0:
            if args.image_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.image_dim # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=-1, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=True # TODO: ProtoCPC and DINO needs this layer
                ).to(args.device)
        else:
            self.text_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Text backbone do not append projection head, so set args.joint_projection_dim = self.text_dim')
            args.joint_projection_dim = self.text_dim

        # image projection head
        if args.image_head_n_layers>0:
            if args.text_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.text_dim # adaption layer
            self.image_projection_head = DINOHead(
                in_dim=self.image_dim, out_dim=-1, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.image_head_n_layers, skip_last_layer=True # TODO: ProtoCPC and DINO needs this layer
                ).to(args.device)
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        self.logit_scale = self.text_backbone.logit_scale if hasattr(self.text_backbone, 'logit_scale') else torch.autograd.Variable(torch.ones(1) * np.log(1 / 0.07)).to(self.device)
        self.logit_scale = nn.Parameter(self.logit_scale)
        self.logit_scale.requires_grad = True
        self.to(self.device)


    def encode_image(self, images, projection=False):
        image_features = self.image_backbone(images)
        if projection:
            image_features = self.image_projection_head(image_features)
        return image_features

    def encode_text(self, texts, projection=False):

        if self.text_model_builder=='OpenCLIP':
            texts = self.tokenizer(texts, context_length=77).to(self.device)
            def open_clip_forward(texts):
                x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model]
                # if prompt is not None:
                #     batch_prompt = prompt().unsqueeze(0).expand(x.size(0), -1, -1)
                #     x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
                x = x + self.text_backbone.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.text_backbone.ln_final(x) # [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                return x

            text_features = open_clip_forward(texts)
            if projection:
                text_features = self.text_projection_head(text_features)
            return text_features
        
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
            if projection:
                text_features = self.text_projection_head(text_features)
            return text_features
    
    def forward(self, images, texts):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        images: torch.tensor (batchs_size, token_indexs)
        """
        image_features = self.encode_image(images, projection=True)
        text_features = self.encode_text(texts, projection=True)

        return image_features, text_features, self.logit_scale.exp()
        

