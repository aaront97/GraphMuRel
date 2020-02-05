import torch


def soft_cross_entropy(x, correct_indices, weight_indices, reduce='mean'):
    """
    Input: 
        x -- [batch size x no_probs] no_probs is expected to be log_probs
    If not, please apply LogSoftmax() before this loss function
        correct_indices -- LongTensor of correct indices
        weight_indices -- Tensor of indices weights

    Keyword Arguments:
        reduce -- can be mean or sum
    Output:
        Soft Cross Entropy Loss, log_probs * weight_indices
        https://ilija139.github.io/pub/cvpr2017_vqa.pdf
    """
    x = x.gather(1, correct_indices)
    x = x * weights
    x = x.sum(dim=1)
    x = torch.mean(x)
    return x