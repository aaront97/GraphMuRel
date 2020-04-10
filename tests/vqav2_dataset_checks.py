# -*- coding: utf-8 -*-
import pytest
from dataset.VQAv2Dataset import VQAv2Dataset
import yaml
from torch.utils.data import DataLoader
import os
import re


def standardise(s):
    s = s.lower()
    s = s.rstrip()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        s = re.sub(i, '', s)
    return s


def tokenize(self, s):
    # we don't replace # because # is used to refer to number of items
    # Tokenizing code taken from Cadene
    t_str = standardise(s)

    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)

    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

@pytest.fixture(scope='session')
def config():
    with open('/home/bat34/VQA_PartII/murel/scripts/murel.yaml') as f:
        config = yaml.load(f)
    config['batch_size'] = 32
    return config


@pytest.fixture(scope='session')
def anno(config):
    path_train_anno = os.path.join(config['vqa_dir'],
                                   'Annotations',
                                   'v2_mscoco_train2014_annotations.json')
    with open(path_train_anno) as f:
        anno = yaml.load(f)
    anno = anno['annotations']
    annotations = {}
    for anno_dict in anno:
        qid = anno_dict['question_id']
        inner_dict = {}
        inner_dict['image_id'] = anno_dict['image_id']
        inner_dict['answers'] = anno_dict['answers']
        inner_dict['multiple_choice_answer'] = anno_dict['multiple_choice_answer']
        annotations[qid] = inner_dict
    return annotations


@pytest.fixture(scope='session')
def ques(config):
    path_train_ques = os.path.join(config['vqa_dir'],
                                   'Questions',
                                   'v2_OpenEnded_mscoco_train2014_questions.json')
    with open(path_train_ques) as f:
        ques = yaml.load(f)

    ques = ques['questions']
    questions = {}
    for ques_dict in ques:
        qid = ques_dict['question_id']
        inner_dict = {}
        inner_dict['image_id'] = ques_dict['image_id']
        inner_dict['question'] = ques_dict['question']
        questions[qid] = inner_dict
    return questions


@pytest.fixture(scope='session')
def dataset(config):
    '''Returns the training dataset'''
    dataset = VQAv2Dataset(split="train",
                           txt_enc=config['txt_enc'],
                           bottom_up_features_dir=config['bottom_up_features_dir'],
                           skipthoughts_dir=config['skipthoughts_dir'],
                           processed_dir=config['processed_dir'],
                           ROOT_DIR=config['ROOT_DIR'],
                           vqa_dir=config['vqa_dir'])
    return dataset


@pytest.fixture(scope='session')
def loader(config, dataset):
    loader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=config['batch_size'],
                        num_workers=config['num_workers'],
                        collate_fn=dataset.collate_fn)
    return loader


def test_legitimate_question_ids(dataset, loader, ques):
    iter_loader = iter(loader)
    batch = next(iter_loader)
    batch_question_unique_ids = batch['question_unique_id']
    for id in batch_question_unique_ids:
        assert id in ques


def test_reconstruct_question(dataset, loader, ques):
    iter_loader = iter(loader)
    batch = next(iter_loader)
    batch_word_ids = batch['question_ids']
    batch_word_ids = list(batch_word_ids)
    batch_word_ids = [[tensor.item() for tensor in l] for l in batch_word_ids]

    reconstructed = []
    for word_list in batch_word_ids:
        inner = []
        for wid in word_list:
            if wid in dataset.wid_to_word:
                inner.append(dataset.wid_to_word[wid])
        reconstructed.append(inner)

    correct_tokens = []
    for qid in batch['question_unique_id']:
        string_qns = ques[qid]['question']
        string_qns_tokenized = tokenize(string_qns)
        correct_tokens.append(string_qns_tokenized)
    assert correct_tokens == reconstructed


def test_question_lengths(dataset, loader, ques):
    iter_loader = iter(loader)
    batch = next(iter_loader)
    batch['question_lengths'] = list(batch['question_lengths'])
    for i, qid in enumerate(batch['question_unique_id']):
        string_qns = ques[qid]['question']
        string_qns_tokenized = tokenize(string_qns)
        assert len(string_qns_tokenized) == batch['question_lengths'][i].item()

