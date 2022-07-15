import torch
import torch.nn as nn
import torch.nn.functional as F

class SimReg(nn.Module):
    def __init__(self, args, dim) -> None:
        super().__init__()
        self.MSELoss = nn.MSELoss()
    
    def forward(self, teacher_feature, student_feature):
        return self.MSELoss(F.normalize(teacher_feature, dim=1), F.normalize(student_feature, dim=1))

class SimRegL1(nn.Module):
    def __init__(self, args, dim) -> None:
        super().__init__()
        self.L1Loss = nn.L1Loss()
    
    def forward(self, teacher_feature, student_feature):
        return self.L1Loss(F.normalize(teacher_feature, dim=1), F.normalize(student_feature, dim=1))

class SimRegSmoothL1(nn.Module):
    def __init__(self, args, dim) -> None:
        super().__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss()
    
    def forward(self, teacher_feature, student_feature):
        return self.SmoothL1Loss(F.normalize(teacher_feature, dim=1), F.normalize(student_feature, dim=1))
