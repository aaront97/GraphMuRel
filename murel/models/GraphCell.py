# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_net.factory.factory_fusion import factory_fusion
from torch_geometric.nn import GCNConv, global_max_pool

class GraphCell(nn.Module):
    def fuse_object_features_with_questions(self,
                                            object_features_list,
                                            question_embedding,
                                            batch_size,
                                            num_obj):

#        res = self.fusion_features([
#            question_embedding,
#            object_features_list.contiguous().view(batch_size * num_obj, -1),
#        ])
        
        res = self.fusion_features([
                question_embedding,
                object_features_list
                ])
        
        #res = res.view(batch_size, num_obj, -1)
        return res
  
    def __init__(self, in_channels, out_channels, config):
        super(GraphCell, self).__init__()
        fusion_features_cfg = config['obj_features_question']
        self.fusion_features = factory_fusion(fusion_features_cfg)
        self.gcn1 = GCNConv(in_channels, in_channels)
#        self.gcn2 = GCNConv(in_channels, in_channels)
#        self.gcn3 = GCNConv(in_channels, out_channels)

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
#        x = F.relu(x)
#        x = self.gcn2(x, edge_index)
#        x = F.relu(x)
#        x = self.gcn3(x, edge_index)
        # x = global_max_pool(x, batch)
        return x
