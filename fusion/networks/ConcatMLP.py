# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
class ConcatMLP(nn.Module):
    def __init__(self, config):
        super(ConcatMLP, self).__init__()
        self.input_dims = config['input_dims']
        self.input_dim = sum(self.input_dims)
        self.out_dim = config['out_dim']
        self.dropout = config['dropout']
        self.hidden_list = config['hidden_list']
        self.hidden = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_list[0])])
        for length1, length2 in zip(self.hidden_list[:-1], self.hidden_list[1:]):
            self.hidden.append(nn.Linear(length1, length2))
        self.last_layer = nn.Linear(self.hidden_list[-1], self.out_dim)
    
    def forward(self, x):
        x = torch.cat(x, dim=1)
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.last_layer(x)
        return x
