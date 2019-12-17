import torch
import torch.nn as nn

# http://juditacs.github.io/2018/12/27/masked-attention.html
# Compute attention weights such that the padded units give 0 attention weights
def masked_softmax(x, lengths):
    max_length = x.size(1)
    indices = torch.arange(max_length).to(device=x.device)
    indices = indices[None, :, None].expand_as(x)
    lengths_expand = lengths.unsqueeze(2).expand_as(x)
    mask = indices < lengths_expand
    x[~mask] = float('-inf')
    x = torch.softmax(x, dim=1)
    return x