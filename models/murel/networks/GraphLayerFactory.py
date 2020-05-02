import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv


class GraphLayerFactory:
    def __init__(self):
        pass

    def get_graph_layer(self, config):
        if config['graph_layer_type'] == 'gcnconv':
            return GCNConv
        elif config['graph_layer_type'] == 'gatconv':
            return GATConv
        elif config['graph_layer_type'] == 'graphconv':
            return GraphConv
        else:
            raise ValueError('Unrecognised graph layer type. Current available layers: GCN, GAT, GraphConv.')
