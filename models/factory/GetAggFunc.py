import torch


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
