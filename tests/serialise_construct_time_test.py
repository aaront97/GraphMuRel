# -*- coding: utf-8 -*-
import progressbar
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import os
import subprocess
import timeit


def constr_graph(feat_dir, images_list, graph_files_dir):
    constructor = T.KNNGraph(k=6, force_undirected=True)
    res = []
    for i in progressbar.progressbar(range(0, len(images_list))):
        name = images_list[i]
        feat_path = os.path.join(feat_dir, name)
        dict_feats = torch.load(feat_path)
        data = Data(x=dict_feats['pooled_feat'], pos=dict_feats['norm_rois'])
        res.append(constructor(data))
    return res


def save_graphs(feat_dir, images_list, graph_files_dir):
    constructor = T.KNNGraph(k=6, force_undirected=True)
    for i in progressbar.progressbar(range(0, len(images_list))):
        name = images_list[i]
        feat_path = os.path.join(feat_dir, name)
        dict_feats = torch.load(feat_path)
        data = Data(x=dict_feats['pooled_feat'], pos=dict_feats['norm_rois'])
        data = constructor(data)
        torch.save(data, os.path.join(graph_files_dir, name))


def load_graphs(feat_dir, images_list, graph_files_dir):
    res = []
    for i in progressbar.progressbar(range(0, len(images_list))):
        name = images_list[i]
        graph_path = os.path.join(graph_files_dir, name)
        data = torch.load(graph_path)
        res.append(data)
    return res



feat_dir = '/auto/homes/bat34/2018-04-27_bottom-up-attention_fixed_36'
graph_files_dir = '/auto/homes/bat34/VQA_PartII/tests/test_graphs'
no_images = 5000

if not os.path.exists(graph_files_dir):
    subprocess.run(['mkdir', '-p', graph_files_dir])

images_list = os.listdir(feat_dir)[:no_images]
def main():
    t = timeit.timeit('constr_graph(feat_dir, images_list, graph_files_dir)',
                      'from __main__ import constr_graph, feat_dir, images_list, graph_files_dir', number=1)
    print('Constructing graphs took an average of: {} s per graph'.format(t / no_images))
    t = timeit.timeit('save_graphs(feat_dir, images_list, graph_files_dir)', 
                      'from __main__ import save_graphs, feat_dir, images_list, graph_files_dir', number=1)
    print('Saving graphs took: {} s'.format(t))
    t = timeit.timeit('load_graphs(feat_dir, images_list, graph_files_dir)',
                      'from __main__ import load_graphs, feat_dir, images_list, graph_files_dir', number=1)
    print('Loading graphs took an average of: {} s per graph'.format(t / no_images))

if __name__ == '__main__':
    main()
