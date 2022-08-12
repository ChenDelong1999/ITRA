import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ce_loss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, pla, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.pla = pla

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = teacher_output / temp
        #teacher_out = teacher_out.detach()
        if self.pla == 'softmax':
            teacher_out = F.softmax(teacher_out, dim=-1)
        elif self.pla == 'sk':
            teacher_out = sk_uniform(teacher_out)
        else:
            raise NotImplementedError()

        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()

class protocpc_loss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()

        out_dim = 65536
        warmup_teacher_temp = 0.04
        teacher_temp = 0.04     
        warmup_teacher_temp_epochs = 0
        nepochs = args.epochs
        #pla = 'sk'
        pla = 'softmax'
        prior_momentum = 0.9         
        student_temp = 0.1

        self.student_temp = student_temp
        self.prior_momentum = prior_momentum
        self.pla = pla
        self.register_buffer("prior", torch.ones(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_output, student_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        
        # teacher centering and sharpening
        epoch = 0
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = teacher_output / temp
        # teacher_out = teacher_out.detach()
        if self.pla == 'softmax':
            teacher_out = F.softmax(teacher_out, dim=-1)
        elif self.pla == 'sk':
            teacher_out = sk_uniform(teacher_out)
        else:
            raise NotImplementedError()


        loss_1 = -torch.sum(teacher_out * student_out, dim=1)
        loss_2 = torch.logsumexp(student_out + torch.log(self.prior), 1)
        loss = loss_1 + loss_2

        self.update_prior(teacher_out)
        return loss.mean()


    @torch.no_grad()
    def update_prior(self, teacher_out):
        batch_prior = torch.sum(teacher_out, dim=0, keepdim=True)
        dist.all_reduce(batch_prior)

        batch_prior = batch_prior / torch.sum(batch_prior)
        batch_prior *= batch_prior.size(1)
        # ema update
        self.prior = self.prior * self.prior_momentum + batch_prior * (1 - self.prior_momentum)

#@torch.no_grad()
def sk_uniform(output, nmb_iters=3):
    Q = torch.exp(output).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * dist.get_world_size() # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(nmb_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()