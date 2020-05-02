# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.factory.FusionFactory import FusionFactory
from torch_geometric.nn import GCNConv
from models.murel.networks.GraphLayerFactory import GraphLayerFactory

class GraphCell(nn.Module):
    def fuse_object_features_with_questions(self,
                                            object_features_list,
                                            question_embedding,
                                            batch_size,
                                            num_obj):

        
        res = self.fusion_features([
                question_embedding,
                object_features_list
                ])

        return res
  
    def __init__(self, config):
        super(GraphCell, self).__init__()
        fusion_factory = FusionFactory()
        graph_layer_factory = GraphLayerFactory()
        fusion_features_cfg = config['fusion']['obj_features_question']
        self.fusion_features = fusion_factory.create_fusion(fusion_features_cfg)
        graph_cfg = config['graph']
        kwargs = graph_cfg['kwargs']
        graph_layer = graph_layer_factory.get_graph_layer(config['graph'])

        self.graph_hidden_list = nn.ModuleList([graph_layer(graph_cfg['input_dim'],
                                                            graph_cfg['graph_hidden_list'][0],
                                                            **kwargs
                                                            )])
        if len(graph_cfg['graph_hidden_list']) > 1:
            for length1, length2 in zip(graph_cfg['graph_hidden_list'][:-1], graph_cfg['graph_hidden_list'][1:]):
                self.graph_hidden_list.append(graph_layer(length1, length2, **kwargs))
        self.last_layer = graph_layer(graph_cfg['graph_hidden_list'][-1], graph_cfg['output_dim'], **kwargs)


    def forward(self,
                question_embedding,
                object_features_list,
                bounding_boxes,
                batch_size,
                num_obj,
                data):
        fused_question_object = self.fuse_object_features_with_questions(
                    object_features_list,
                    question_embedding,
                    batch_size,
                    num_obj)

        x = fused_question_object
        edge_index, batch = data.edge_index, data.batch
        for layer in self.graph_hidden_list:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.last_layer(x, edge_index)
        return x
