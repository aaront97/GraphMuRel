import torch.nn as nn
from models.text_encoders.SkipthoughtsFactory import get_text_enc
from fusion.networks.ConcatMLP import ConcatMLP


class ResnetConcat(nn.Module):
    def __init__(self, config, word_vocabulary):
        super(ResnetConcat, self).__init__()
        self.txt_enc = get_text_enc(config, word_vocabulary)
        self.concat_mlp = ConcatMLP(config['fusion_mlp'])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, item):
        question_each_word_embedding = self.txt_enc.embedding(item['question_ids'])
        question_features, question_final_feature = self.txt_enc.rnn(
                question_each_word_embedding)
        resnet_features = item['resnet_features']

        x = self.concat_mlp([question_final_feature, resnet_features])
        x = self.log_softmax(x)
        return self.log_softmax(x)
