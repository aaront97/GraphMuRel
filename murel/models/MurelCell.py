# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fusion_net.factory.factory_fusion import factory_fusion
from dataset.auxiliary_functions import get_aggregation_func, masked_softmax


class MurelCell(nn.Module):
    def __init__(self, config):
        super(MurelCell, self).__init__()
        fusion_features_cfg = config['fusion']['obj_features_question']
        fusion_box_cfg = config['fusion']['box']
        fusion_fused_cfg = config['fusion']['obj_features_obj_features']
        
        if 'murel_cell_attention' in config:
            self.murel_cell_attention = True
            self.murel_cell_attention_linear0 = nn.Linear(
                    config['murel_cell_attention']['linear0']['input_dim'],
                    config['murel_cell_attention']['linear0']['output_dim'])
            self.murel_cell_attention_linear1 = nn.Linear(
                    config['murel_cell_attention']['linear1']['input_dim'],
                    config['murel_cell_attention']['linear1']['output_dim'])    
        else:
            self.murel_cell_attention = False
        
        self.fusion_features = factory_fusion(fusion_features_cfg)
        self.fusion_box = factory_fusion(fusion_box_cfg)
        self.fusion_fused = factory_fusion(fusion_fused_cfg)
        self.pairwise_agg = get_aggregation_func(config['pairwise_agg'], dim=2)
    
    def compute_relation_attention(self, relations):
        _, _, no_objects, _ = relations.size()
        r_att = self.linear0(relations)
        r_att = torch.nn.functional.tanh(r_att)
        r_att = self.linear1(r_att)

        # http://juditacs.github.io/2018/12/27/masked-attention.html
        # Compute attention weights such that the padded units
        # give 0 attention weights
        r_att = masked_softmax(r_att, no_objects)
        # Glimpses contain attention values for each question_feature
        # DIM: BATCH_SIZE x NO_WORDS
        r_att = r_att.unsqueeze(3).expand(-1, -1, no_objects)
        r_att = relations * r_att
        r_att = torch.sum(r_att, dim=2)
        return r_att

    def pairwise(self, 
                 fused_features, 
                 bounding_boxes, 
                 batch_size, 
                 num_obj):
        relations = self.fusion_fused(
                [
                        fused_features.unsqueeze(2).expand(-1, -1, num_obj, -1)
                        .contiguous().view(batch_size * num_obj * num_obj, -1),
                        fused_features.unsqueeze(1).contiguous().expand(-1, num_obj, -1, -1)
                        .contiguous().view(batch_size * num_obj * num_obj, -1)
                ]
                ) + \
                    self.fusion_box(
                [
                        bounding_boxes.unsqueeze(2).expand(-1, -1, num_obj, -1)
                        .contiguous().view(batch_size * num_obj * num_obj, -1),
                        bounding_boxes.unsqueeze(1).contiguous().expand(-1, num_obj, -1, -1)
                        .contiguous().view(batch_size * num_obj * num_obj, -1)
                ]
                )
        relations = relations.view(batch_size, num_obj, num_obj, -1)
        if self.murel_cell_attention:
            e_hat = self.compute_relation_attention(relations)
        else:
            e_hat = self.pairwise_agg(relations)
        res = fused_features + e_hat
        return res

    def fuse_object_features_with_questions(self,
                                            object_features_list,
                                            question_embedding,
                                            batch_size,
                                            num_obj):

        res = self.fusion_features([
            question_embedding,
            object_features_list.contiguous().view(batch_size * num_obj, -1),
        ])
        res = res.view(batch_size, num_obj, -1)
        return res

    def forward(self,
                question_embedding,
                object_features_list,
                bounding_boxes,
                batch_size,
                num_obj):
        # Sensitive?
        fused_question_object = self.fuse_object_features_with_questions(
                object_features_list,
                question_embedding,
                batch_size,
                num_obj)

        pairwise_res = self.pairwise(
                fused_question_object,
                bounding_boxes,
                batch_size,
                num_obj)
        res = object_features_list + pairwise_res
        return res
