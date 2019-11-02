# -*- coding: utf-8 -*-
from skipthoughts import BayesianUniSkip, UniSkip, BiSkip, DropUniSkip
def get_text_enc(skipthoughts_dir, text_enc, vocab):
    if text_enc == 'BayesianUniSkip':
        return BayesianUniSkip(skipthoughts_dir, vocab)
    if text_enc == 'UniSkip':
        return UniSkip(skipthoughts_dir, vocab)
    if text_enc == 'BiSkip':
        return BiSkip(skipthoughts_dir, vocab)
    if text_enc == 'DropUniSkip':
        return DropUniSkip(skipthoughts_dir, vocab)
        
