# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool


class GraphCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.gcn1 = GCNConv(in_channels, in_channels)
        self.gcn2 = GCNConv(in_channels, out_channels)

    def forward(self, data):
        batch = data.batch
        data = self.gcn1(data.x, data.edge_index)
        data = F.relu(data.x)
        data = self.gcn2(data.x, data.edge_index)
        data = global_max_pool(data.x, batch)
        return data
