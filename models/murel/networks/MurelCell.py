# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fusion.factory.FusionFactory import FusionFactory
from models.factory.GetAggFunc import get_aggregation_func
from transforms.transforms import masked_softmax


class MurelCell(nn.Module):
    def __init__(self, config):
        super(MurelCell, self).__init__()
        self.fusion_factory = FusionFactory()
        fusion_features_cfg = config['fusion']['obj_features_question']
        fusion_box_cfg = config['fusion']['box']
        fusion_fused_cfg = config['fusion']['obj_features_obj_features']
        
        if config['murel_attention']:
            self.murel_cell_attention = True
            self.murel_cell_attention_linear0 = nn.Linear(
                    config['murel_cell_attention']['linear0']['input_dim'],
                    config['murel_cell_attention']['linear0']['output_dim'])
            self.murel_cell_attention_linear1 = nn.Linear(
                    config['murel_cell_attention']['linear1']['input_dim'],
                    config['murel_cell_attention']['linear1']['output_dim'])    
        else:
            self.murel_cell_attention = False

        self.buffer = None
        self.fusion_features = self.fusion_factory.create_fusion(fusion_features_cfg)
        self.fusion_box = self.fusion_factory.create_fusion(fusion_box_cfg)
        self.fusion_fused = self.fusion_factory.create_fusion(fusion_fused_cfg)
        self.pairwise_agg = get_aggregation_func(config['pairwise_agg'], dim=2)

    def initialise_buffers(self):
        self.buffer = {}

    def compute_relation_attention(self, relations):
        _, _, no_objects, no_feats = relations.size()
        r_att = self.murel_cell_attention_linear0(relations)
        r_att = torch.nn.functional.tanh(r_att)
        r_att = self.murel_cell_attention_linear1(r_att)
        r_att = torch.softmax(r_att, dim=2)
        r_att = torch.squeeze(torch.unbind(r_att, dim=3)[0])
        r_att = r_att.unsqueeze(3).expand(-1, no_objects, no_objects, no_feats)
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
        if self.buffer is not None:
            _, argmax = torch.max(res, dim=1)
            self.buffer['argmax'] = argmax.data.cpu()
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
