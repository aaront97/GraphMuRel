# -*- coding: utf-8 -*-
from dataset.AbstractVQADataset import AbstractVQADataset
import torch
import os
from dataset.TextEncFactory import get_text_enc
import transforms.transforms as transforms
class MuralNetDataset(AbstractVQADataset):
    def __init__(self, \
              bottom_up_features_dir, \
              object_features_dir='/local/scratch/bat34/', \
              split='train', \
              ROOT_DIR='/auto/homes/bat34/VQA_PartII/', \
              txt_enc='BayesianUniSkip',\
              collate_fn=None):
        #Change this#########
        self.collate_fn = transforms.Compose([ \
                transforms.ConvertBatchListToDict(), \
                transforms.CreateBatchItem() \
                ]) if collate_fn is None else collate_fn
        ############
        self.bottom_up_features_dir = bottom_up_features_dir
        self.object_features_dir = object_features_dir
        self.split = split
        skipthoughts_dir = os.path.join(ROOT_DIR, 'data', 'skipthoughts')
        self.text_enc = get_text_enc(skipthoughts_dir, txt_enc, [word for key, word in \
                                                                 self.wid_to_word.items()])
        if split == 'train':
            self.dataset = self.train_set
        if split == 'val':
            self.dataset = self.val_set
        if split == 'test':
            self.dataset = self.test_set
    
    def __getitem__(self, idx):
        item = {}
        question = self.dataset['questions'][idx]
        item['index'] = idx
        item['question_id'] = question['question_id']
        item['image_name'] = question['image_name']
        question_ids = torch.LongTensor([question['question_ids']])
        image_name = question['image_name']
        question_vector = self.text_enc(question_ids, [len(question_ids)])
        question_vector = torch.squeeze(question_vector)
        dict_features = torch.load(image_name + '.pth')
        item['bounding_boxes'] = dict_features['norm_rois']
        item['object_features_list'] = dict_features['pooled_feat']
        item['question_embedding'] = question_vector
        if self.split != 'test':
            annotation = self.dataset['annotations'][idx]
            item['answer_id'] = torch.LongTensor([annotation['answer_id']])
            item['answer'] = annotation['most_frequent_answer']
            item['question_type'] = annotation['question_type']
        return item