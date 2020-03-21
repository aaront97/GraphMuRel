import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.auxiliary_functions import masked_softmax
from dataset.TextEncFactory import get_text_enc
from baseline.models.ConcatMLP import ConcatMLP
from fusion_net.factory.factory_fusion import factory_fusion


class AttentionNet(nn.Module):
    def __init__(self, config, word_vocabulary):
        super(AttentionNet, self).__init__()

        if config['attention_fusion_type'] == 'concat_mlp':
            self.attention_fusion = ConcatMLP(config['attention_fusion_mlp'])
        elif config['attention_fusion_type'] == 'block':
            self.attention_fusion = factory_fusion(
                    config['attention_fusion_block'])
        else:
            raise ValueError('Unimplemented attention fusion')

        if config['final_fusion_type'] == 'concat_mlp':
            self.final_fusion = ConcatMLP(config['final_fusion_mlp'])
        elif config['final_fusion_type'] == 'block':
            self.final_fusion = factory_fusion(config['final_fusion_block'])
        else:
            raise ValueError('Unimplemented final fusion')

        self.txt_enc = get_text_enc(config, word_vocabulary)
        self.q_linear0 = nn.Linear(
                config['q_att']['q_linear0']['input_dim'],
                config['q_att']['q_linear0']['output_dim'])
        self.q_linear1 = nn.Linear(
                config['q_att']['q_linear1']['input_dim'],
                config['q_att']['q_linear1']['output_dim'])

        self.obj_linear0 = nn.Linear(
                config['obj_att']['obj_linear0']['input_dim'],
                config['obj_att']['obj_linear0']['output_dim'])
        self.obj_linear1 = nn.Linear(
                config['obj_att']['obj_linear1']['input_dim'],
                config['obj_att']['obj_linear1']['output_dim'])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, item):
        question_ids = item['question_ids']
        object_features_list = item['object_features_list']
        question_lengths = item['question_lengths']

        question_each_word_embedding = self.txt_enc.embedding(question_ids)
        question_features, question_final_feature = self.txt_enc.rnn(
                question_each_word_embedding)

        question_attentioned = self.self_attention_question(
                question_features, question_lengths)

        object_attentioned = self.compute_object_attention_with_question(
                question_attentioned, object_features_list)

        # Construct training vector
        x = self.final_fusion([question_attentioned, object_attentioned])
        x = self.log_softmax(x)
        return x
 
    def compute_object_attention_with_question(self, question_self_attentioned, object_features_list):
        batch_size = object_features_list.size(0)
        no_objects = object_features_list.size(1)
        q_expanded = question_self_attentioned.unsqueeze(1).expand(-1, no_objects, -1)
        fused = self.attention_fusion(
                [
                 q_expanded.contiguous().view(batch_size * no_objects, -1),
                 object_features_list.contiguous().view(batch_size * no_objects, -1)
                ]
        )
        fused = fused.view(batch_size, no_objects, -1)
        fused_att = self.obj_linear0(fused)
        fused_att = F.relu(fused_att)
        fused_att = self.obj_linear1(fused_att)
        fused_att = F.softmax(fused_att, dim=1)

        glimpses = torch.unbind(fused_att, dim=2)
        attentioned_glimpses = []
        for glimpse in glimpses:
            glimpse = glimpse.unsqueeze(2).expand(-1, -1, object_features_list.size(-1))
            attentioned_feature = object_features_list * glimpse
            attentioned_feature = torch.sum(attentioned_feature, dim=1)
            attentioned_glimpses.append(attentioned_feature)
        fused_attentioned = torch.cat(attentioned_glimpses, dim=1)
        return fused_attentioned

    def self_attention_question(self, question_features, question_lengths):
        q_att = self.q_linear0(question_features)
        q_att = torch.nn.functional.relu(q_att)
        q_att = self.q_linear1(q_att)

        # http://juditacs.github.io/2018/12/27/masked-attention.html
        # Compute attention weights such that the padded units give
        # 0 attention weights
        q_att = masked_softmax(q_att, question_lengths)
        # Glimpses contain attention values for each question_feature
        # DIM: BATCH_SIZE x NO_WORDS
        glimpses = torch.unbind(q_att, dim=2)
        attentioned_glimpses = []
        for glimpse in glimpses:
            glimpse = glimpse.unsqueeze(2).expand(-1, -1, question_features.size(-1))
            attentioned_feature = question_features * glimpse
            attentioned_feature = torch.sum(attentioned_feature, dim=1)
            attentioned_glimpses.append(attentioned_feature)
        question_attentioned = torch.cat(attentioned_glimpses, dim=1)
        return question_attentioned   
