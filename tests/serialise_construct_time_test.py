# -*- coding: utf-8 -*-
import progressbar
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import os
import subprocess
import time
feat_dir = '/auto/homes/bat34/2018-04-27_bottom-up-attention_fixed_36'
graph_files_dir = '/auto/homes/bat34/VQA_PartII/tests/test_graphs'
no_images = 1000

if not os.path.exists(test_files_dir):
    subprocess.run(['mkdir', '-p', test_files_dir])

images_list = os.listdir(feat_dir)[:no_images]


def constr_graph(feat_dir, images_list, graph_files_dir):
    constructor = T.KNNGraph(k=6, force_undirected=True)
    res = []
    for name in images_list:
        feat_path = feat_dir + name
        dict_feats = torch.load(feat_path)
        data = Data(x=dict_feats['pooled_feat'], pos=dict_feats['norm_rois'])
        res.append(constructor(data))
    return res


def save_graphs(feat_dir, images_list, graph_files_dir):
    constructor = T.KNNGraph(k=6, force_undirected=True)
    for name in images_list:
        feat_path = feat_dir + name
        dict_feats = torch.load(feat_path)
        data = Data(x=dict_feats['pooled_feat'], pos=dict_feats['norm_rois'])
        torch.save(data, os.path.join(graph_files_dir, name))

     
def load_graphs(feat_dir, images_list, graph_files_dir):
    res = []
    for name in images_list:
        graph_path = os.path.join(graph_files_dir, name)
        data = torch.load(graph_path)
        res.append(data)
    return res


def main():
    

if __name__ == '__main__':
    main()
        