from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import random 
from PIL import ImageFilter



def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        augmentation=None
):
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        if not augmentation:
            return Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])

        elif augmentation == 'protoclip-light-augmentation':
            s = 1
            size = image_size
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            gaussian_blur = transforms.GaussianBlur(kernel_size=21)
            return Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([gaussian_blur], p=0.2),
                transforms.ToTensor(),
                normalize
                ])

        elif augmentation == 'SLIP':
            class GaussianBlur(object):
                """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

                def __init__(self, sigma=[.1, 2.]):
                    self.sigma = sigma

                def __call__(self, x):
                    sigma = random.uniform(self.sigma[0], self.sigma[1])
                    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
                    return x

            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                _convert_to_rgb,
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
