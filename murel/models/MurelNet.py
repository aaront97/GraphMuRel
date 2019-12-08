# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import fusions
from fusion_net.factory.factory_fusion import factory_fusion, fusion_types
from murel.models.MurelCell import MurelCell

class MurelNet(nn.Module):
    def __init__(self, config):
        super(MurelNet, self).__init__()
        self.murel_cell = MurelCell(config['fusion'])
        self.final_fusion = factory_fusion(config['fusion']['final_input_fusion_dims'], \
                                           config['fusion']['final_output_fusion_dims'], \
                                           config['fusion']['final_fusion_type'], \
                                           config['fusion']['dropout_final'])
        self.unroll_steps = config['unroll_steps']
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, item):
        question_embedding = item['question_embedding']
        object_features_list = item['object_features_list']
        bounding_boxes = item['bounding_boxes']
        for i in range(self.unroll_steps):
            object_features_list = self.murel_cell(question_embedding ,\
                                                   object_features_list, \
                                                   bounding_boxes)
        max_pool, _ = torch.max(object_features_list, dim=0)
        scores = self.final_fusion([max_pool, question_embedding])
        prob = self.log_softmax(scores)
        return prob
