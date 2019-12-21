import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.auxiliary_functions import masked_softmax
from dataset.TextEncFactory import get_text_enc

class ConcatBaselineNet(nn.Module):
    #Todo Dropout?
    def __init__(self, config, word_vocabulary):
        #Hidden list expects a list of arguments that specifies
        #The depth and length of the hidden layers, e.g.
        # [4448, 1024, 512, 3000]
        super(ConcatBaselineNet, self).__init__()
        self.input_dim = config['input_dim']
        self.out_dim = config['out_dim']
        self.dropout = config['dropout']
        self.hidden_list = config['hidden_list']
        self.hidden = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_list[0])])
        self.txt_enc = get_text_enc(config, word_vocabulary)
        self.q_linear0 = nn.Linear(config['q_att']['q_linear0']['input_dim'], \
                                 config['q_att']['q_linear0']['output_dim'])
        self.q_linear1 = nn.Linear(config['q_att']['q_linear1']['input_dim'], \
                                 config['q_att']['q_linear1']['output_dim'])
        
        self.obj_linear0 = nn.Linear(config['obj_att']['obj_linear0']['input_dim'], \
                                 config['obj_att']['obj_linear0']['output_dim'])
        self.obj_linear1 = nn.Linear(config['obj_att']['obj_linear1']['input_dim'], \
                                 config['obj_att']['obj_linear1']['output_dim'])
        
        for length1, length2 in zip(self.hidden_list[:-1], self.hidden_list[1:]):
            self.hidden.append(nn.Linear(length1, length2))
        self.last_layer = nn.Linear(self.hidden_list[-1], self.out_dim)

    def forward(self, item):
        question_ids = item['question_ids']
        object_features_list = item['object_features_list']
        question_lengths = item['question_lengths']
        
        question_each_word_embedding = self.txt_enc.embedding(question_ids)
        question_features, question_final_feature = self.txt_enc.rnn(question_each_word_embedding)
        
        question_attentioned = self.self_attention_question(question_features, question_lengths)
        object_attentioned = self.self_attention_object(object_features_list)
        
        # Construct training vector
        x = torch.cat([question_attentioned, object_attentioned], dim=1)
        
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x)
        x = self.last_layer(x)
        return x
    
    def self_attention_object(self, object_features):
        obj_att = self.obj_linear0(object_features)
        obj_att = torch.nn.functional.relu(obj_att)
        obj_att = self.obj_linear1(obj_att)
        
        obj_att = torch.nn.functional.softmax(obj_att, dim=1)
        glimpses = torch.unbind(obj_att, dim=2)
        attentioned_glimpses = []
        for glimpse in glimpses:
            glimpse = glimpse.unsqueeze(2).expand(-1, -1, object_features.size(-1))
            attentioned_feature = object_features * glimpse
            attentioned_feature = torch.sum(attentioned_feature, dim=1)
            attentioned_glimpses.append(attentioned_feature)
        object_attentioned = torch.cat(attentioned_glimpses, dim=1)
        return object_attentioned
        
    
    
    def self_attention_question(self, question_features, question_lengths):
        q_att = self.q_linear0(question_features)
        q_att = torch.nn.functional.relu(q_att)
        q_att = self.q_linear1(q_att)

        # http://juditacs.github.io/2018/12/27/masked-attention.html
        # Compute attention weights such that the padded units give 0 attention weights
        q_att = masked_softmax(q_att, question_lengths)
        # Glimpses contain attention values for each question_feature DIM: BATCH_SIZE x NO_WORDS
        glimpses = torch.unbind(q_att, dim=2)
        attentioned_glimpses = []
        for glimpse in glimpses:
            glimpse = glimpse.unsqueeze(2).expand(-1, -1, question_features.size(-1))
            attentioned_feature = question_features * glimpse
            attentioned_feature = torch.sum(attentioned_feature, dim=1)
            attentioned_glimpses.append(attentioned_feature)
        question_attentioned = torch.cat(attentioned_glimpses, dim=1)
        return question_attentioned

        