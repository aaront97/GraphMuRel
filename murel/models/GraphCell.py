# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool


class GraphCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphCell, self).__init__()
        self.gcn1 = GCNConv(in_channels, in_channels)
        self.gcn2 = GCNConv(in_channels, in_channels)
        self.gcn3 = GCNConv(in_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)
        x = global_max_pool(x, batch)
        return x
