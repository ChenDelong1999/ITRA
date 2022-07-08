import torch
import torch.nn as nn


def get_projection_head(input_dim, args):
    if args.projection_n_layers==0:
        return nn.Sequential(nn.Linear(input_dim, args.projection_dim)).to(args.device)
    else:
        layers = [
            nn.Linear(input_dim, args.projection_hidden_dim),
            nn.BatchNorm1d(args.projection_hidden_dim),
            nn.ReLU(inplace=False)]
        for i in range(args.projection_n_layers-1):
            layers.extend([
                nn.Linear(args.projection_hidden_dim, args.projection_hidden_dim),
                nn.BatchNorm1d(args.projection_hidden_dim), 
                nn.ReLU(inplace=False)])
        layers.append(nn.Linear(args.projection_hidden_dim, args.projection_dim))
        return nn.Sequential(*layers).to(args.device)


def add_projection_head(model, input_dim, args):
    # all distiller needs image projection head
    model.image_projection_head = get_projection_head(input_dim, args)
    if args.add_teacher_projection_head:
        model.text_projection_head = get_projection_head(args.projection_dim, args)

        if args.add_teacher_projection_AE:
            model.text_decoder = get_projection_head(args.projection_dim, args)


    return model