import torch
import torch.nn as nn

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """
    def __init__(self, args, dim):
        """
        dim:        feature dimension (default: 128)
        K:          queue size
        t:          temperature for student encoder
        # temp:       distillation temperature
        # base_width: width of the base network
        # swav_mlp:   MLP length for SWAV resnet, default=None
        """
        super(SEED, self).__init__()
        
        #dim=args.projection_dim
        self.K = 65536
        self.t = 0.07
        self.temp = 1e-4
        self.dim = dim
        self.dist = args.distributed

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        # self.student = student(num_classes=dim)

        # if not swav_mlp:
        #     self.teacher = teacher(num_classes=dim, width_multiplier=base_width)
        # else:
        #     self.teacher = teacher(normalize=True, hidden_mlp=swav_mlp, output_dim=dim)

        # if mlp:
        #     dim_mlp = self.student.fc.weight.shape[1]
        #     self.student.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.fc)

        #     if stu == 'moco':
        #         dim_mlp = self.teacher.fc.weight.shape[1]
        #         self.teacher.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher.fc)

        # # not update by gradient
        # for param_k in self.teacher.parameters():
        #     param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, teacher_feature, student_feature):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """

        # # compute query features
        # s_emb = self.student(image)  # NxC
        # s_emb = nn.functional.normalize(s_emb, dim=1)

        # # compute key features
        # with torch.no_grad():  # no gradient to keys

        #     t_emb = self.teacher(image)  # keys: NxC
        #     t_emb = nn.functional.normalize(t_emb, dim=1)

        t_emb = nn.functional.normalize(teacher_feature, dim=1)
        s_emb = nn.functional.normalize(student_feature, dim=1)

        # cross-Entropy Loss
        logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])
        logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])

        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

        logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
        logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

        # compute soft labels
        logit_stu /= self.t
        logit_tea = nn.functional.softmax(logit_tea/self.temp, dim=1)

        # de-queue and en-queue
        self._dequeue_and_enqueue(t_emb, concat=self.dist)

        return -(logit_tea * torch.nn.functional.log_softmax(logit_stu, 1)).sum()/logit_stu.shape[0]

