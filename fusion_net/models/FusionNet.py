# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_net.factory.factory_fusion import factory_fusion, fusion_types
import block

class FusionNet(nn.Module):
    def __init__(self, config, fusion_type=None):
        super(FusionNet, self).__init__()
        self.input_fusion_dims = config['input_fusion_dims']
        self.output_fusion_dims = config['output_fusion_dims']
        fusion_type = fusion_type if fusion_type else config['fusion_type']
        self.fusion = factory_fusion(self.input_fusion_dims, self.output_fusion_dims, \
                                     fusion_type, config['dropout'])
        self.input_dim = self.output_fusion_dims
        self.output_dim = config['output_dim']
        self.hidden = nn.ModuleList()
        if not hidden_list or len(hidden_list) <= 2:
            self.hidden.append(nn.LayerNorm(input_dim))
            self.hidden.append(nn.Linear(input_dim, out_dim))
        else:
            assert hidden_list[0] == input_dim and hidden_list[-1] == out_dim, \
             "Please make the first and last element equal to input_dim and out_dim respectively"
            for i in range(len(hidden_list) - 2):
                self.hidden.append(nn.LayerNorm(hidden_list[i]))
                self.hidden.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
                self.hidden.append(nn.Dropout(p=self.dropout))
            self.hidden.append(nn.LayerNorm(hidden_list[-2]))
            self.hidden.append(nn.Linear(hidden_list[-2], hidden_list[-1]))
        
        
    def forward(self, x):
        x = self.fusion(x)
        for i in range(len(self.hidden) - 1):
            layer = self.hidden[i]
            if isinstance(layer, nn.LayerNorm) or isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        #No activation at last layer
        x = self.hidden[-1](x)
        return x