# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fusion_net.factory.factory_fusion import factory_fusion
from murel.models.MurelCell import MurelCell

class MurelNet(nn.Module):
    def __init__(self, config):
        super(MurelNet, self).__init__()
        self.murel_cell = MurelCell(config['fusion'])
        self.final_fusion = factory_fusion(config['fusion']['final_fusion'])
        self.unroll_steps = config['unroll_steps']
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, item):
        question_embedding = item['question_embedding']
        object_features_list = item['object_features_list']
        bounding_boxes = item['bounding_boxes']
        if len(list(object_features_list.size())) != 3:
            raise ValueError('Expecting Object Features Input of BATCH_SIZE x NUM_OBJ x 2048')
        
        batch_size, num_obj, _ = list(object_features_list.size())
        #Resize question outside for loop as it would be used repeatedly in multiple unroll steps
        #Reshape question (BATCH_SIZE x QUES_DIM) TO (BATCH_SIZE x NUM_OBJ x QUES_DIM)
        question_embedding_repeated = question_embedding.unsqueeze(1).expand(-1, num_obj, -1).contiguous()
        
        #Reshape question to (BATCH_SIZE * NUM_OBJ x QUES_DIM)
        question_embedding_repeated = question_embedding_repeated.view(batch_size * num_obj, -1)
        
        
        for i in range(self.unroll_steps):
            object_features_list = self.murel_cell(question_embedding_repeated ,\
                                                   object_features_list, \
                                                   bounding_boxes, \
                                                   batch_size, \
                                                   num_obj)
        max_pool, _ = torch.max(object_features_list, dim=1)
        scores = self.final_fusion([max_pool, question_embedding])
        prob = self.log_softmax(scores)
        return prob
