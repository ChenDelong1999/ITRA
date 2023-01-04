import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, text_features, image_features, logit_scale=2.659, labels=None):
        image_loss = self.cross_entropy(image_features, labels)
        text_loss = self.cross_entropy(text_features, labels)

        total_loss = (image_loss + text_loss) / 2

        return total_loss



class UniCLLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()

    def forward(self, text_features, image_features, logit_scale=2.659, labels=None):

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        target_matrixs = torch.eye(num_logits, device=device, dtype=torch.float)
        for i in range(num_logits):
            label = labels[i]
            if label!=-1: # samples from image-pair dataset have no possitive pairs other than itself
                possitives = (label==labels).float() # N
                target_matrixs[i, :] = possitives
                target_matrixs[:, i] = possitives

        # total_loss = (
        #     F.cross_entropy(logits_per_image, target_matrixs) +
        #     F.cross_entropy(logits_per_text, target_matrixs)
        #     ) / 2

        total_loss = (
            self.SoftCE(logits_per_image, target_matrixs) +
            self.SoftCE(logits_per_text, target_matrixs)
            ) / 2

        return total_loss
    
    def SoftCE(self, s, t):
        s = torch.softmax(s, dim=-1)
        loss = - (t * torch.log(s)).sum(dim=-1)
        return (loss/t.sum(dim=-1)).mean()


