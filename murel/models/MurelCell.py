# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fusion_net.factory.factory_fusion import factory_fusion
from dataset.auxiliary_functions import get_aggregation_func


class MurelCell(nn.Module):
    def __init__(self, config):
        super(MurelCell, self).__init__()
        fusion_features_cfg = config['obj_features_question']
        fusion_box_cfg = config['box']
        fusion_fused_cfg = config['obj_features_obj_features']

        self.fusion_features = factory_fusion(fusion_features_cfg)
        self.fusion_box = factory_fusion(fusion_box_cfg)
        self.fusion_fused = factory_fusion(fusion_fused_cfg)
        self.pairwise_agg = get_aggregation_func(config['pairwise_agg'], dim=2)

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
