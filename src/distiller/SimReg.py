import torch
import torch.nn as nn
import torch.nn.functional as F

class SimReg(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.MSE = nn.MSELoss()
    
    def forward(self, teacher_feature, student_feature):
        return self.MSE(F.normalize(teacher_feature, dim=1), F.normalize(student_feature, dim=1))

class SimRegL1(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.L1 = nn.L1Loss()
    
    def forward(self, teacher_feature, student_feature):
        return self.L1Loss(F.normalize(teacher_feature, dim=1), F.normalize(student_feature, dim=1))