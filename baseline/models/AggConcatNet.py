import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.auxiliary_functions import masked_softmax
from dataset.TextEncFactory import get_text_enc
from baseline.models.ConcatMLP import ConcatMLP
from fusion_net.factory.factory_fusion import factory_fusion


class AggConcatNet(nn.Module):
    def __init__(self, config, word_vocabulary):
        super(AggConcatNet, self).__init__()

        self.agg_type = config['agg_type']
        self.q_self_attention = config['q_self_attention']
        if config['fusion_type'] == 'concat_mlp':
            self.fusion = ConcatMLP(config['fusion_mlp'])
        elif config['fusion_type'] == 'block':
            self.fusion = factory_fusion(
                    config['fusion_block'])
        else:
            raise ValueError('Unimplemented attention fusion')

        self.txt_enc = get_text_enc(config, word_vocabulary)
        self.q_linear0 = nn.Linear(
                config['q_att']['q_linear0']['input_dim'],
                config['q_att']['q_linear0']['output_dim'])
        self.q_linear1 = nn.Linear(
                config['q_att']['q_linear1']['input_dim'],
                config['q_att']['q_linear1']['output_dim'])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, item):
        question_ids = item['question_ids']
        object_features_list = item['object_features_list']
        question_lengths = item['question_lengths']

        question_each_word_embedding = self.txt_enc.embedding(question_ids)
        question_features, question_final_feature = self.txt_enc.rnn(
                question_each_word_embedding)

        if self.q_self_attention:
            question_attentioned = self.self_attention_question(
                    question_features, question_lengths)
        else:
            question_attentioned = question_final_feature

        object_attentioned = self.process_butd_features(object_features_list,
                                                        self.agg_type)

        # Construct training vector
        x = self.fusion([question_attentioned, object_attentioned])
        x = self.log_softmax(x)
        return x
    
    def process_butd_features(self, object_features, agg_type):
        if agg_type == 'mean':
            return torch.mean(object_features, dim=1)
        if agg_type == 'max':
            return torch.max(object_features, dim=1)[0]
        if agg_type == 'min':
            return torch.min(object_features, dim=1)[0]
        if agg_type == 'sum':
            return torch.sum(object_features, dim=1)

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
