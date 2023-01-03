import torch
import torch.nn as nn
from .SimReg import SimReg, SimRegL1, SimRegSmoothL1
from .VICReg import VICReg
from .BarlowTwins import BarlowTwins
from .RKD import RKD
from .CompRess import CompReSS, CompReSSA
from .CLIP import CLIPLoss
from .UniCL import UniCLLoss
from .DINO import DINOLoss
from .SEED import SEED
from .ProtoCPC import protocpc_loss
from .ProtoRKD import ProtoRKDLoss

NEED_LOGIT_SCALE = ['InfoNCE', 'UniCL']
NEED_GATHER = ['InfoNCE', 'UniCL']
UNI_DIRECTIONAL = ['CompRess-1q', 'SEED', 'DINO', 'ProtoCPC']
NEED_PROTOTYPE_LAYER = ['DINO', 'ProtoCPC', 'ProtoRKD']


def get_loss(args):
    if args.loss=='SimReg':
        return SimReg
    elif args.loss=='SimReg-L1':
        return SimRegL1
    elif args.loss=='SimReg-SmoothL1':
        return SimRegSmoothL1
    elif args.loss=='VICReg':
        return VICReg
    elif args.loss=='BarlowTwins':
        return BarlowTwins
    elif args.loss=='RKD':
        return RKD
    elif args.loss=='CompRess-2q':
        return CompReSS
    elif args.loss=='CompRess-1q':
        return CompReSSA
    elif args.loss=='SEED':
        return SEED
    elif args.loss=='InfoNCE':
        return CLIPLoss
    elif args.loss=='UniCL':
        return UniCLLoss
    elif args.loss=='DINO':
        return DINOLoss
    elif args.loss=='ProtoCPC':
        return protocpc_loss
    elif args.loss=='ProtoRKD':
        return ProtoRKDLoss


if __name__=='__main__':
    rkd_loss = RKD(None)
    teacher_feature, student_feature = torch.randn([4,256]), torch.randn([4,256])
    print(student_feature)
    print(student_feature.size())
    print()

    loss = rkd_loss(teacher_feature, student_feature)
    print(loss)
