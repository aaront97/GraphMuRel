import yaml
from models.murel.networks.MurelNet import MurelNet
from models.baseline.networks.AttentionNet import AttentionNet
from models.baseline.networks.AggConcatNet import AggConcatNet


class ModelFactory:
    def create_model(self, model_name, config, word_vocabulary):
        if model_name == 'murel':
            return MurelNet(config, word_vocabulary)
        elif model_name == 'attention':
            return AttentionNet(config, word_vocabulary)
        elif model_name == 'frcnn_concat':
            return AggConcatNet(config, word_vocabulary)
        elif model_name == 'resnet_concat':
            raise ValueError('Resnet Concat API not implemented')
        else:
            raise ValueError()


    def create_config(self, model_name):
        if model_name == 'murel':
            with open('murel/configs/murel.yaml') as f:
                config = yaml.load(f)
            return config
        elif model_name == 'attention':
            with open('baseline/configs/attention_baseline.yaml') as f:
                config = yaml.load(f)
            return config
        elif model_name == 'frcnn_concat':
            with open('baseline/configs/agg_baseline.yaml') as f:
                config = yaml.load(f)
            return config
        elif model_name == 'resnet_concat':
            raise ValueError('Resnet Concat API not implemented')
        else:
            raise ValueError()
