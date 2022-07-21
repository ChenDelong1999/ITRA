import torch

class Prompt(torch.nn.Module):
    def __init__(self, n_context, n_dim, args) -> None:
        super().__init__()
        
        from torch.autograd import Variable
        self.prompt = torch.empty(n_context, n_dim)
        torch.nn.init.normal_(self.prompt, std=0.02)
        self.prompt = Variable(self.prompt).to(args.device)
        self.prompt.requires_grad = True

    def forward(self):
        return self.prompt


def encode_text_with_prompt(clip, tokens, prompt=None):
    x = clip.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
    if prompt is not None:
        batch_prompt = prompt().unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
    x = x + clip.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip.transformer(x, attn_mask=clip.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip.ln_final(x) # [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ clip.text_projection
    return x