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
        
    def pairwise(self, fused_features, bounding_boxes, batch_size, num_obj):
        relations = self.fusion_fused(
                        [ \
                                fused_features.unsqueeze(2).expand(-1, -1, num_obj, -1) \
                                .contiguous().view(batch_size * num_obj * num_obj, -1),\
                                fused_features.unsqueeze(1).contiguous().expand(-1, num_obj, -1, -1) \
                                .contiguous().view(batch_size * num_obj * num_obj, -1) \
                        ] \
                        ) + \
                    self.fusion_box(
                        [ \
                                bounding_boxes.unsqueeze(2).expand(-1, -1, num_obj, -1) \
                                .contiguous().view(batch_size * num_obj * num_obj, -1),\
                                bounding_boxes.unsqueeze(1).contiguous().expand(-1, num_obj, -1, -1) \
                                .contiguous().view(batch_size * num_obj * num_obj, -1) \
                        ] \
                        ) 
        relations = relations.view(batch_size, num_obj, num_obj, -1)
        e_hat, _ = torch.max(relations, dim=2)
        res = fused_features + e_hat
        return res
    
    def fuse_object_features_with_questions(self, object_features_list, question_embedding, batch_size, num_obj):
        res =  self.fusion_features([ \
                    object_features_list.contiguous().view(batch_size * num_obj, -1), \
                    question_embedding
                ])
        res = res.view(batch_size, num_obj, -1)
        return res
    
    def forward(self, question_embedding, object_features_list, bounding_boxes, batch_size, num_obj):
        fused_question_object = self.fuse_object_features_with_questions(object_features_list, question_embedding, \
                                                          batch_size, num_obj)
        pairwise_res = self.pairwise(fused_question_object, bounding_boxes, batch_size, num_obj)
        res = object_features_list + pairwise_res
        return res