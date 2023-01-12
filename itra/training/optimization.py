import torch
from torch import optim as optim
import logging

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

from training.distributed import is_master

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

# from https://github.com/LightDXY/FT-CLIP/blob/main/optim_factory.py

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("class_embedding", "cls_token", "mask_token", "pos_embed", "positional_embedding"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith("conv1"):
        return 0
    elif var_name.startswith("ln_pre"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("image_backbone.transformer.resblocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1

    else:
        return num_max_layer - 1


def get_num_layer_for_resnet(var_name, num_max_layer):

    if var_name.startswith('image_backbone.layer'):
        layer_id = int(var_name.split('.')[1].replace('layer',''))
        return layer_id + 1 # index start from 1 instead of 0 (as in ViT's resblocks), and stem block account for 1 layer
    elif var_name.startswith('image_backbone.attnpool'):
        return num_max_layer -1
    else:
        # stem
        return 1

def get_num_layer_for_text_transformer(var_name, num_max_layer):
    if var_name in ("class_embedding", "cls_token", "mask_token", "pos_embed", "positional_embedding"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith("conv1"):
        return 0
    elif var_name.startswith("ln_pre"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith(f"text_backbone.transformer.resblocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1

    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        if layer_id is not None:
            return self.values[layer_id]
        else:
            return 1.

    def get_layer_id(self, var_name, backbone):
        if backbone=='ResNet':
            return get_num_layer_for_resnet(var_name, len(self.values))
        if backbone=='ViT':
            return get_num_layer_for_vit(var_name, len(self.values))
        elif backbone=='text_backbone':
            return get_num_layer_for_text_transformer(var_name, len(self.values))
        else:
            raise RuntimeError(f'Failed to get layer id for {var_name}, backbone={backbone}')


def get_parameter_groups(args, model, weight_decay=1e-5, skip_list=(), get_num_layer_image=None, get_num_layer_text=None, get_layer_scale_image=None, get_layer_scale_text=None, return_name=False):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'relative_position_bias' in name:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        layer_id = None
        if 'image' in name:
            if get_num_layer_image is not None:
                layer_id = get_num_layer_image(name, backbone=model.image_backbone.arch)
                group_name = "image_layer_%d_%s" % (layer_id, group_name)
        if 'text' in name:
            if get_num_layer_text is not None:
                layer_id = get_num_layer_text(name, backbone='text_backbone')
                group_name = "text_layer_%d_%s" % (layer_id, group_name)            

        if group_name not in parameter_group_names:
            if 'image' in name and get_layer_scale_image is not None:
                scale = get_layer_scale_image(layer_id)
            elif 'text' in name and get_layer_scale_text is not None:
                scale = get_layer_scale_text(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
    if is_master(args):
        logging.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    if return_name:
        return list(parameter_group_vars.values()), parameter_group_names
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer_image=None, get_num_layer_text=None, get_layer_scale_image=None, get_layer_scale_text=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(args, model, weight_decay, skip, get_num_layer_image, get_num_layer_text, get_layer_scale_image, get_layer_scale_text)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        print ("USING Optimizer:", opt_lower, opt_args)
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

def create_adamw_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    optimizer = optim.AdamW(parameters, lr=args.lr, 
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def get_optimizer(model, args):
    model_without_ddp = model.module if args.distributed else model
    if args.layer_decay_image < 1.0:
        num_layers_image = model_without_ddp.image_backbone.layers
        if is_master(args):
            logging.info(f'Image backbone has {num_layers_image} layers')
        decay = list(args.backbone_decay * args.layer_decay_image ** (num_layers_image + 1 - i) for i in range(num_layers_image + 2))
        decay[-1] /= args.backbone_decay
        assigner_image = LayerDecayValueAssigner(decay)
    else:
        assigner_image = None
        
    if args.layer_decay_text < 1.0:
        num_layers_text = model_without_ddp.text_backbone.layers
        if is_master(args):
            logging.info(f'Text backbone has {num_layers_text} layers')
        decay = list(args.backbone_decay * args.layer_decay_text ** (num_layers_text + 1 - i) for i in range(num_layers_text + 2))
        decay[-1] /= args.backbone_decay
        assigner_text = LayerDecayValueAssigner(decay)
    else:
        assigner_text = None

    # TODO
    # skip_weight_decay_list = model.no_weight_decay()
    skip_weight_decay_list = {'positional_embedding', 'class_embedding', 'logit_scale', 'bn', 'ln', 'bias'}
    
    # TODO
    # args.disable_weight_decay_on_rel_pos_bias = False 
    # if args.disable_weight_decay_on_rel_pos_bias:
    #     for i in range(num_layers):
    #         skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    return create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer_image=assigner_image.get_layer_id if assigner_image is not None else None, 
            get_num_layer_text=assigner_text.get_layer_id if assigner_text is not None else None, 
            get_layer_scale_image=assigner_image.get_scale if assigner_image is not None else None,
            get_layer_scale_text=assigner_text.get_scale if assigner_text is not None else None,
            )