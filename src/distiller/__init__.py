import torch
import torch.nn as nn
from .SimReg import SimReg, SimRegL1
from .RKD import RKD
from .CompRess import CompReSS, CompReSSA
from .CLIP import CLIPLoss


def get_distiller(distiller):
    if distiller=='SimReg':
        return SimReg
    elif distiller=='SimReg-L1':
        return SimRegL1
    elif distiller=='RKD':
        return RKD
    elif distiller=='CompRess':
        return CompReSS
    elif distiller=='CompRess-1q':
        return CompReSSA
    elif distiller=='CLIP':
        return CLIPLoss


if __name__=='__main__':
    rkd_loss = RKD(None)
    teacher_feature, student_feature = torch.randn([4,256]), torch.randn([4,256])
    print(student_feature)
    print(student_feature.size())
    print()

    loss = rkd_loss(teacher_feature, student_feature)
    print(loss)
