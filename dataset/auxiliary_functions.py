import torch

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


def get_aggregation_func(agg_type, dim):
    if agg_type == 'max':
        def f(x):
            return torch.max(x, dim=dim)[0]
        return f
    if agg_type == 'min':
        def f(x):
            return torch.min(x, dim=dim)[0]
        return f
    if agg_type == 'sum':
        def f(x):
            return torch.sum(x, dim=dim)
        return f
    if agg_type == 'mean':
        def f(x):
            return torch.mean(x, dim=dim)
        return f
    
def tokenize(self, s):
    # we don't replace # because # is used to refer to number of items
    # Tokenizing code taken from Cadene
    s = s.rstrip()
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub(i, '', t_str)

    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list