# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fusion_net.factory.factory_fusion import factory_fusion
from murel.models.MurelCell import MurelCell
from murel.models.GraphCell import GraphCell
from dataset.TextEncFactory import get_text_enc
from dataset.auxiliary_functions import masked_softmax, get_aggregation_func


class MurelNet(nn.Module):
    def __init__(self, config, word_vocabulary):
        super(MurelNet, self).__init__()
        self.murel_cell = MurelCell(config['fusion'])
        self.final_fusion = factory_fusion(config['fusion']['final_fusion'])
        self.unroll_steps = config['unroll_steps']
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.txt_enc = get_text_enc(config, word_vocabulary)
        self.linear0 = nn.Linear(config['q_att']['linear0']['input_dim'],
                                 config['q_att']['linear0']['output_dim'])
        self.linear1 = nn.Linear(config['q_att']['linear1']['input_dim'],
                                 config['q_att']['linear1']['output_dim'])
        self.pooling_agg = get_aggregation_func(config['pooling_agg'], dim=1)
        self.include_graph_module = config['include_graph_module']

        if self.include_graph_module:
            self.graph_module = GraphCell(config['graph']['input_dim'],
                                          config['graph']['output_dim'])
            self.graph_fusion = factory_fusion(
                    config['fusion']['graph_fusion'])

    def forward(self, item):
        question_ids = item['question_ids']
        object_features_list = item['object_features_list']
        bounding_boxes = item['bounding_boxes']
        question_lengths = item['question_lengths']
        graph_batch = item['graph_batch']


        # q_att
        question_each_word_embedding = self.txt_enc.embedding(question_ids)
        question_features, question_final_feature = self.txt_enc.rnn(
                question_each_word_embedding)

        q_att = self.linear0(question_features)
        q_att = torch.nn.functional.relu(q_att)
        q_att = self.linear1(q_att)

        # http://juditacs.github.io/2018/12/27/masked-attention.html
        # Compute attention weights such that the padded units
        # give 0 attention weights
        q_att = masked_softmax(q_att, question_lengths)
        # Glimpses contain attention values for each question_feature
        # DIM: BATCH_SIZE x NO_WORDS
        glimpses = torch.unbind(q_att, dim=2)
        attentioned_glimpses = []
        for glimpse in glimpses:
            glimpse = glimpse.unsqueeze(2).expand(-1,
                                                  -1,
                                                  question_features.size(-1))
            attentioned_feature = question_features * glimpse
            attentioned_feature = torch.sum(attentioned_feature, dim=1)
            attentioned_glimpses.append(attentioned_feature)
        question_attentioned = torch.cat(attentioned_glimpses, dim=1)

        if len(list(object_features_list.size())) != 3:
            raise ValueError(
                    'Expecting Object Features Input' +
                    'of BATCH_SIZE x NUM_OBJ x 2048')

        batch_size, num_obj, _ = list(object_features_list.size())
        # Resize question outside for loop as
        # it would be used repeatedly in multiple unroll steps
        # Reshape question
        # (BATCH_SIZE x QUES_DIM) TO (BATCH_SIZE x NUM_OBJ x QUES_DIM)
        question_attentioned_repeated = question_attentioned.unsqueeze(1).expand(
                -1, num_obj, -1).contiguous()

        # Reshape question to (BATCH_SIZE * NUM_OBJ x QUES_DIM)
        question_attentioned_repeated = question_attentioned_repeated.view(
                batch_size * num_obj, -1)

        for i in range(self.unroll_steps):
            object_features_list = self.murel_cell(
                                    question_attentioned_repeated,
                                    object_features_list,
                                    bounding_boxes,
                                    batch_size,
                                    num_obj)
        pool = self.pooling_agg(object_features_list)
        if self.include_graph_module:
            graph_res = self.graph_module(graph_batch)
            pool = self.graph_fusion([pool, graph_res])
        scores = self.final_fusion([question_attentioned, pool])
        prob = self.log_softmax(scores)
        return prob
