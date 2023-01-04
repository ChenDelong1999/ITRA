import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class Cacher():
    def __init__(self, n_sample, n_dim, cache_file) -> None:
        self.feature = torch.zeros(size=(n_sample, n_dim))
        self.cache_file = cache_file

    def load_batch(self, index, features):
        self.feature[index] = features.detach().cpu()

    def save(self):
        np.save(self.cache_file, self.feature)