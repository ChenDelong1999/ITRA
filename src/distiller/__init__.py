import torch
import torch.nn as nn
from .SimReg import SimReg, SimRegL1, SimRegSmoothL1
from .RKD import RKD
from .CompRess import CompReSS, CompReSSA
from .CLIP import CLIPLoss
from .DINO import DINOLoss
from .SEED import SEED
from .ProtoCPC import protocpc_loss


def get_distiller(distiller):
    if distiller=='SimReg':
        return SimReg
    elif distiller=='SimReg-L1':
        return SimRegL1
    elif distiller=='SimReg-SmoothL1':
        return SimRegSmoothL1
    elif distiller=='RKD':
        return RKD
    elif distiller=='CompRess-2q':
        return CompReSS
    elif distiller=='CompRess-1q':
        return CompReSSA
    elif distiller=='SEED':
        return SEED
    elif distiller=='InfoNCE':
        return CLIPLoss
    elif distiller=='DINO':
        return DINOLoss
    elif distiller=='ProtoCPC':
        return protocpc_loss


if __name__=='__main__':
    rkd_loss = RKD(None)
    teacher_feature, student_feature = torch.randn([4,256]), torch.randn([4,256])
    print(student_feature)
    print(student_feature.size())
    print()

    loss = rkd_loss(teacher_feature, student_feature)
    print(loss)
