import torch
import torch.nn as nn
import torch.nn.functional as F

class RKD(nn.Module):
    '''
    https://github.com/megvii-research/mdistiller/blob/255c16fc32882a697bfd35569307380090562b2c/mdistiller/distillers/RKD.py#L21
    
    '''
    def __init__(self, args, dim) -> None:
        super().__init__()    
        self.squared = False
        self.args = args
        
    def _pdist(self, e, squared, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        
        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
        
    def forward(self, teacher_feature, student_feature):
        teacher_feature = F.normalize(teacher_feature, dim=1)
        student_feature = F.normalize(student_feature, dim=1)
        
        # Distance loss
        if self.args.w_rkd_d!=0:
            t_d = self._pdist(teacher_feature, self.squared)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
            
            d = self._pdist(student_feature, self.squared)
            mean_d = d[d > 0].mean()
            d = d / mean_d

            loss_distance = F.smooth_l1_loss(d, t_d)
        else:
            loss_distance = 0

        # Angle loss
        if self.args.w_rkd_a!=0:
            td = (teacher_feature.unsqueeze(0) - teacher_feature.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            sd = (student_feature.unsqueeze(0) - student_feature.unsqueeze(1))
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
            loss_angle = F.smooth_l1_loss(s_angle, t_angle)
        else:
            loss_angle = 0

        
        return self.args.w_rkd_d * loss_distance + self.args.w_rkd_a * loss_angle


