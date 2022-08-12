import torch
import torch.nn as nn
from .SimReg import SimReg, SimRegL1, SimRegSmoothL1
from .VICReg import VICReg
from .BarlowTwins import BarlowTwins
from .RKD import RKD
from .CompRess import CompReSS, CompReSSA
from .CLIP import CLIPLoss
from .DINO import DINOLoss
from .SEED import SEED
from .ProtoCPC import protocpc_loss

NEED_LOGIT_SCALE = ['InfoNCE']
NEED_GATHER = ['InfoNCE']
UNI_DIRECTIONAL = ['CompRess-1q', 'SEED', 'DINO', 'ProtoCPC']
NEED_PROTOTYPE_LAYER = ['DINO', 'ProtoCPC']


def get_distiller(args):
    if args.distiller=='SimReg':
        return SimReg
    elif args.distiller=='SimReg-L1':
        return SimRegL1
    elif args.distiller=='SimReg-SmoothL1':
        return SimRegSmoothL1
    elif args.distiller=='VICReg':
        return VICReg
    elif args.distiller=='BarlowTwins':
        return BarlowTwins
    elif args.distiller=='RKD':
        return RKD
    elif args.distiller=='CompRess-2q':
        return CompReSS
    elif args.distiller=='CompRess-1q':
        return CompReSSA
    elif args.distiller=='SEED':
        return SEED
    elif args.distiller=='InfoNCE':
        return CLIPLoss
    elif args.distiller=='DINO':
        return DINOLoss
    elif args.distiller=='ProtoCPC':
        return protocpc_loss


if __name__=='__main__':
    rkd_loss = RKD(None)
    teacher_feature, student_feature = torch.randn([4,256]), torch.randn([4,256])
    print(student_feature)
    print(student_feature.size())
    print()

    loss = rkd_loss(teacher_feature, student_feature)
    print(loss)
