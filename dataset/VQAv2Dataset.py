# -*- coding: utf-8 -*-
from dataset.AbstractVQADataset import AbstractVQADataset
import torch
import os
import transforms.transforms as transforms


class VQAv2Dataset(AbstractVQADataset):
    def __init__(self,
                 bottom_up_features_dir='',
                 split='train',
                 ROOT_DIR='/auto/homes/bat34/VQA_PartII/',
                 txt_enc='BayesianUniSkip',
                 collate_fn=None,
                 processed_dir='/auto/homes/bat34/VQA_PartII/data/processed_splits',
                 model='murel',
                 vqa_dir='/auto/homes/bat34/VQA',
                 no_answers=3000,
                 sample_answers=False,
                 skipthoughts_dir='/auto/homes/bat34/VQA_PartII/data/skipthoughts',
                 include_graph_data=True,
                 graph_type='knn6',
                 resnet_features='/local/scratch/bat34/resnet101-features-2048'
                 ):
        super(VQAv2Dataset, self).__init__(
                 processed_dir=processed_dir,
                 model="murel",
                 vqa_dir=vqa_dir,
                 no_answers=no_answers,
                 sample_answers=sample_answers,
                 skipthoughts_dir=skipthoughts_dir,
                 split=split)
        self.resnet_features_dir = resnet_features
        self.bottom_up_features_dir = bottom_up_features_dir
        self.split = split
        self.include_graph_data = include_graph_data


        if graph_type.startswith('knn'):
            no_neigh = int(graph_type.lstrip('knn'))
            print('You have picked nearest-neighbour graphs, with N = {}'.format(no_neigh))
            self.graph_dir = '/local/scratch/bat34/graphs' + \
                             'graphs/preprocessed_graphs_knn_neighbours_{}/'.format(no_neigh)

        if self.split == 'train':
            self.collate_fn = transforms.Compose([
                transforms.ConvertBatchListToDict(),
                transforms.Pad1DTensors(dict_keys=['question_ids']),
                #transforms.Pad1DTensors(dict_keys=['question_ids', 'id_unique', 'id_weights']),
                transforms.BatchGraph(),
                transforms.StackTensors(),
                ]) if collate_fn is None else collate_fn
        else:
            self.collate_fn = transforms.Compose([
                transforms.ConvertBatchListToDict(),
                transforms.Pad1DTensors(dict_keys=['question_ids']),
                transforms.BatchGraph(),
                transforms.StackTensors(),
                ]) if collate_fn is None else collate_fn

    def __len__(self):
        return len(self.dataset['questions'])

    def __getitem__(self, idx):
        item = {}
        question = self.dataset['questions'][idx]
        item['index'] = idx
        item['question_unique_id'] = question['question_id']
        item['question_ids'] = torch.LongTensor(question['question_ids'])
        item['question_lengths'] = torch.LongTensor(
                [len(question['question_ids'])])
        item['image_name'] = question['image_name']
        image_name = question['image_name']
        dict_features = torch.load(
                os.path.join(self.bottom_up_features_dir, image_name) + '.pth')
        #resnet_feat = torch.load(
        #        os.path.join(self.resnet_features_dir, image_name) + '.pth')
        #item['resnet_features'] = torch.squeeze(resnet_feat)
        item['bounding_boxes'] = dict_features['norm_rois']
        item['object_features_list'] = dict_features['pooled_feat']
        if self.graph_dir:
            graph_img_name = os.path.join(self.graph_dir, image_name + '.pth')
            item['graph'] = torch.load(graph_img_name)
        if self.split != 'test':
            annotation = self.dataset['annotations'][idx]
            item['answer_id'] = torch.LongTensor([annotation['answer_id']])
            item['answer'] = annotation['most_frequent_answer']
            item['question_type'] = annotation['question_type']
        #if self.split == 'train':
        #    item['id_weights'] = annotation['id_weights']
        #    item['id_unique'] = annotation['id_unique']
        return item
