import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICReg(nn.Module):
    def __init__(self, args, dim) -> None:
        super().__init__()
        self.MSELoss = nn.MSELoss()
        
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        self.num_features = dim
    
    def forward(self, teacher_feature, student_feature):
        
        batch_size = teacher_feature.size(0)
        x = F.normalize(teacher_feature, dim=1)
        y = F.normalize(student_feature, dim=1)
        
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss
