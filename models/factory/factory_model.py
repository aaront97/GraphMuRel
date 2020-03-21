import yaml
from models.murel.networks.MurelNet import MurelNet
from models.baseline.networks.AttentionNet import AttentionNet
from models.baseline.networks.AggConcatNet import AggConcatNet


def factory_model(model_name, word_vocabulary):
    if model_name == 'murel':
        with open('murel/configs/murel.yaml') as f:
            config = yaml.load(f)
        return MurelNet(config, word_vocabulary)
    elif model_name == 'attention':
        with open('baseline/configs/attention_baseline.yaml') as f:
            config = yaml.load(f)
        return AttentionNet(config, word_vocabulary)
    elif model_name == 'agg_concat':
        with open('baseline/configs/agg_baseline.yaml') as f:
            config = yaml.load(f)
        return AggConcatNet(config, word_vocabulary)
    elif model_name == 'resnet_concat':
        raise ValueError('Resnet Concat API not implemented')
    else:
        raise ValueError()
