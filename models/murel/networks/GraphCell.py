# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.factory.FusionFactory import FusionFactory
from torch_geometric.nn import GCNConv

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
        self.fusion_factory = FusionFactory()
        fusion_features_cfg = config['fusion']['obj_features_question']
        self.fusion_features = self.fusion_factory.create_fusion(fusion_features_cfg)
        graph_cfg = config['graph']
        self.gcn1 = GCNConv(graph_cfg['gcn1']['input_dim'],
                            graph_cfg['gcn1']['output_dim'])
        self.gcn2 = GCNConv(graph_cfg['gcn2']['input_dim'],
                            graph_cfg['gcn2']['output_dim'])
        self.gcn3 = GCNConv(graph_cfg['gcn3']['input_dim'],
                            graph_cfg['gcn3']['output_dim'])

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
        # (batch_size x num_obj x obj_features) -->
        # (batch_size * num_obj x obj_features)
#        fused_question_object = fused_question_object.view(
#                batch_size * num_obj, -1)
        x = fused_question_object
        edge_index, batch = data.edge_index, data.batch
        x = self.gcn1(x, edge_index)
        # x = x + fused_question_object
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        # x = x + fused_question_object
        x = F.relu(x)
        x = self.gcn3(x, edge_index)
        return x
