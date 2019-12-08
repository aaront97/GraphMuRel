# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import fusions
from fusion_net.factory.factory_fusion import factory_fusion, fusion_types

class MurelCell(nn.Module):
    def __init__(self, config):
        super(MurelCell, self).__init__()
        self.fusion_features = factory_fusion(config['features_input_fusion_dims'], \
                                     config['features_output_fusion_dims'], \
                                     config['features_fusion_type'], \
                                     config['dropout_features'])
        self.fusion_box = factory_fusion(config['box_input_fusion_dims'], \
                                         config['box_output_fusion_dims'], \
                                         config['box_fusion_type'], \
                                         config['dropout_box'])
        self.fusion_fused = factory_fusion(config['fused_input_fusion_dims'], \
                                         config['fused_output_fusion_dims'], \
                                         config['fused_fusion_type'], \
                                         config['dropout_fused'])
        
    def pairwise(self, fused_features, bounding_boxes):
        relations = []
        processed = [None for _ in range(len(bounding_boxes))]
        for i, b_i in enumerate(bounding_boxes):
            edges_incoming_i = []
            for j, b_i in enumerate(bounding_boxes):
                if i != j:
                    edges_incoming_i.append(self.fusion_box([bounding_boxes[i], bounding_boxes[j]]) + \
                                self.fusion_fused([fused_features[i], fused_features[j]]))
            relations.append(edges_incoming_i)
                    
        for i in range(len(bounding_boxes)):
            edges = torch.stack(relations[i])
            e_i_hat, _ = torch.max(edges, dim=0)
            processed[i] = fused_features[i] + e_i_hat
        return processed
                            
        
    def forward(self, question_embedding, object_features_list, bounding_boxes):
        #Expect x to be a list of question_vec, and object_vec
        fused_features = []
        for obj_feature in object_features_list:
            fused_features.append(self.fusion_features([obj_feature, question_embedding]))
        additions = self.pairwise(fused_features, bounding_boxes)
        for i in range(len(object_features_list)):
            object_features_list[i] = torch.sum(object_features_list[i], additions[i])
        return object_features_list