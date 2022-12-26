import torch
import torch.nn as nn
import torch.nn.functional as F

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class BarlowTwins(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.lambd = 0.0051

    def forward(self, teacher_feature, student_feature):
        
        batch_size = teacher_feature.size(0)
        z1 = F.normalize(teacher_feature, dim=1)
        z2 = F.normalize(student_feature, dim=1)
        z1 = F.normalize(z1, dim=0)
        z2 = F.normalize(z2, dim=0)

        # empirical cross-correlation matrix
        #c = self.bn(z1).T @ self.bn(z2)
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss