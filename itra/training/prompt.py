import torch
import torch.nn as nn
from torch.autograd import Variable

class Prompt(torch.nn.Module):
    def __init__(self, n_context, n_dim, args) -> None:
        super().__init__()
        
        self.n_prompt = n_context
        self.prompt = torch.empty(n_context, n_dim)
        torch.nn.init.normal_(self.prompt, std=0.02)
        #self.prompt = Variable(self.prompt)
        self.prompt = nn.Parameter(self.prompt).to(args.device)
        #self.prompt.requires_grad = True

    def forward(self):
        return self.prompt

