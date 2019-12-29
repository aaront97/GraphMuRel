import torch
from torch.utils.data import Dataset
from collections import Counter
import os 
import progressbar
import json
import re
import subprocess
import numpy as np


class AbstractVQADataset(Dataset):
    def __init__(self, \
        processed_dir='/auto/homes/bat34/VQA_PartII/data/processed_splits', \
        model='baseline',\
        vqa_dir='/auto/homes/bat34/VQA',\
        no_answers=3000,\
        sample_answers=False,\
        skipthoughts_dir='/auto/homes/bat34/VQA_PartII/data/skipthoughts', 
        split='train', \
        exclude_unk_words=True):
        self.split = split
        self.no_answers = no_answers
        self.skipthoughts_dir = skipthoughts_dir
        self.vqa_dir = vqa_dir
        self.processed_dir = processed_dir
        self.sample_answers = sample_answers
        self.exclude_unk_words = exclude_unk_words
        
        if self.sample_answers:
            self.processed_dir = os.path.join(self.processed_dir, 'sample_answers')
        
        if not os.path.exists(self.processed_dir):
            print('Processing....')
            self.process()
        if split == 'train':
            self.dataset = torch.load(os.path.join(self.processed_dir, 'train2014_processed.pth'))
        if split == 'val':
            self.dataset = torch.load(os.path.join(self.processed_dir, 'val2014_processed.pth'))
        if split == 'test':
            self.dataset = self.test_set = torch.load(os.path.join(self.processed_dir, 'test2015_processed.pth'))
        self.wid_to_word = torch.load(os.path.join(self.processed_dir, 'wid_to_word.pth'))
        self.word_to_wid = torch.load(os.path.join(self.processed_dir, 'word_to_wid.pth'))
        self.ans_to_aid = torch.load(os.path.join(self.processed_dir, 'ans_to_aid.pth'))
        self.aid_to_ans = torch.load(os.path.join(self.processed_dir, 'aid_to_ans.pth'))
    

    def tokenize(self, s):
        # we don't replace # because # is used to refer to number of items
        # Tokenizing code taken from Cadene
        s = s.rstrip()
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub(i, '', t_str)

        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
            q_list = re.sub(r'\?','',t_str.lower()).split(' ')
            q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def merge_questions_with_annotations(self, ques, anno):
        for q_key in ques:
            if q_key not in anno:
                anno[q_key] = ques[q_key]
        return anno

    def get_top_answers(self, split_set):
        #We only get the top most frequent answers because some answers occur only once
        annotations = split_set['annotations']
        counter = {}
        for annotation in annotations:
            counter[annotation['most_frequent_answer']] = \
                counter.get(annotation['most_frequent_answer'], 0) + 1
        sorted_counter = sorted([(word, count) for word, count in counter.items()], \
            key=lambda x: x[1], reverse=True)
        answer_vocabulary = [word[0] for word in sorted_counter[:self.no_answers]]
        print('Length of answer_vocabulary: {}, Original no. of answers: {}'.format(len(answer_vocabulary), len(sorted_counter)))
        return answer_vocabulary

    def get_skipthoughts_dictionary(self):
        dictionary_path = os.path.join(self.skipthoughts_dir, 'dictionary.txt')
        if not os.path.exists(dictionary_path):
            os.system('wget {} -P {}'.format('http://www.cs.toronto.edu/~rkiros/models/dictionary.txt', self.skipthoughts_dir))
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            dict_list = f.read().splitlines()
        return set(dict_list)

    def get_known_words(self, split_set):
        #Concern ourselves with words that the pretrained embedding can embed
        skipthoughts_dictionary = self.get_skipthoughts_dictionary()
        questions_list = split_set['questions']
        known_words_list = []
        unknown_words_list = []
        for question_dict in questions_list:
            for word in question_dict['question_tokens']:
                if word in skipthoughts_dictionary:
                    known_words_list.append(word)
                else:
                    unknown_words_list.append(word)
        known_words_list.append('UNK')
        print('No. of known words: {}, No. of unknown words : {}, Percentage Loss of words: {}%' \
            .format(len(known_words_list), len(unknown_words_list), (len(unknown_words_list) / (len(known_words_list) + len(unknown_words_list))) * 100))
        return list(set(known_words_list)), list(set(unknown_words_list))
    
    def return_all_words(self, split_set):
        print('Disregarding unknown words by the skipthoughts dictionary')
        known_words_list = []
        questions_list = split_set['questions']
        for question_dict in questions_list:
            for word in question_dict['question_tokens']:
                known_words_list.append(word)
        known_words_list = set(known_words_list)
        return known_words_list

    def add_most_frequent_answer(self, split_set):
        for annotation in split_set['annotations']:
            annotation['most_frequent_answer'] = annotation['multiple_choice_answer']
        return split_set

    def add_sampled_answer(self, split_set):
        for annotation in split_set['annotations']:
            counter = {}
            for answer in annotation['answers']:
                answer_str = answer['answer']
                if answer_str not in counter:
                    counter[answer_str] = 0
                counter[answer_str] += 1
            answer_str_list = list(counter.keys())
            answer_str_occurences = [counter[answer_str] for answer_str in answer_str_list]
            answer_str_probs = answer_str_occurences / np.sum(answer_str_occurences)
            annotation['most_frequent_answer'] = np.random.choice(answer_str_list, p=answer_str_probs)
        return split_set
                
                
    def add_images(self, split_set):
        for question in split_set['questions']:
            question['image_name'] = 'COCO_%s_%012d.jpg'%(split_set['data_subtype'],question['image_id'])
        return split_set

    def tokenize_questions(self, split_set):
        print('Tokenizing questions for {}'.format(split_set['data_subtype']))
        for i in progressbar.progressbar(range(len(split_set['questions']))):
            question = split_set['questions'][i]
            question['question_tokens'] = self.tokenize(question['question'])
        return split_set

    def replace_unknown_words_with_UNK(self, split_set, word_to_wid):
        questions_list = split_set['questions']
        for question_dict in questions_list:
            question_dict['question_tokens_UNK'] = \
                [word for word in question_dict['question_tokens'] if word in word_to_wid]
        return split_set

    def add_question_ids(self, split_set, word_to_wid):
        questions_list = split_set['questions']
        for question_dict in questions_list:
            question_dict['question_ids'] = [word_to_wid[word] for word in question_dict['question_tokens_UNK']]
        return split_set

    def remove_question_if_not_top_answer(self, split_set, ans_to_aid):
        print('Removing questions if they have infrequent answers')
        new_annotations = []
        new_questions = []
        if(len(split_set['questions']) != len(split_set['annotations'])):
            raise ValueError()
        for i in progressbar.progressbar(range(len(split_set['annotations']))):
            annotation = split_set['annotations'][i]
            question = split_set['questions'][i]
            if annotation['most_frequent_answer'] in ans_to_aid:
                new_annotations.append(annotation)
                new_questions.append(question)
        split_set['annotations'], split_set['questions'] = new_annotations, new_questions
        return split_set

    def train_encode_answers(self, split_set, ans_to_aid):
        annotations_list = split_set['annotations']
        for annotation_dict in annotations_list:
            annotation_dict['answer_id'] = ans_to_aid[annotation_dict['most_frequent_answer']]
        return split_set

    def val_encode_answers(self, split_set, ans_to_aid):
        annotations_list = split_set['annotations']
        for annotation_dict in annotations_list:
            #For validation, since we are not guaranteed that we have seen the answer before,
            #Set the answer id to something that cannot be predicted by the network.
            annotation_dict['answer_id'] = \
                ans_to_aid.get(annotation_dict['most_frequent_answer'], len(annotation_dict))
        return split_set


    def process(self):
        path_train_anno = os.path.join(self.vqa_dir, 'Annotations', 'v2_mscoco_train2014_annotations.json')
        path_val_anno = os.path.join(self.vqa_dir, 'Annotations', 'v2_mscoco_val2014_annotations.json')
        path_train_ques = os.path.join(self.vqa_dir, 'Questions', 'v2_OpenEnded_mscoco_train2014_questions.json')
        path_val_ques = os.path.join(self.vqa_dir, 'Questions', 'v2_OpenEnded_mscoco_val2014_questions.json')
        path_test_ques = os.path.join(self.vqa_dir, 'Questions', 'v2_OpenEnded_mscoco_test2015_questions.json')

        with open(path_train_ques, 'r') as train_ques_handle, open(path_train_anno, 'r') as train_anno_handle:
            train_set = self.merge_questions_with_annotations(json.load(train_ques_handle), \
                json.load(train_anno_handle))

        with open(path_val_ques, 'r') as val_ques_handle, open(path_val_anno, 'r') as val_anno_handle:
            val_set = self.merge_questions_with_annotations(json.load(val_ques_handle), \
                json.load(val_anno_handle))

        #We don't have annotations for the test set
        with open(path_test_ques, 'r') as test_ques_handle:
            test_set = json.load(test_ques_handle)

        train_set = self.add_images(train_set)
        val_set = self.add_images(val_set)
        test_set = self.add_images(test_set)
        
        if not self.sample_answers:
            train_set = self.add_most_frequent_answer(train_set)
            val_set = self.add_most_frequent_answer(val_set)
        else:
            train_set = self.add_sampled_answer(train_set)
            val_set = self.add_sampled_answer(val_set)

        train_set = self.tokenize_questions(train_set)
        val_set = self.tokenize_questions(val_set)
        test_set = self.tokenize_questions(test_set)

        #We only deal with the top #no_answers
        aid_to_ans = self.get_top_answers(train_set)
        ans_to_aid = {word: index for index, word in enumerate(aid_to_ans)}
        
        if self.exclude_unk_words:
            known_words, _ = self.get_known_words(train_set)
        else:
            known_words = self.return_all_words(train_set)
        wid_to_word = {idx: word for idx, word in enumerate(known_words)}
        word_to_wid = {word:idx for idx, word in enumerate(known_words)}

        train_set = self.remove_question_if_not_top_answer(train_set, ans_to_aid)

        train_set = self.replace_unknown_words_with_UNK(train_set, word_to_wid)
        val_set = self.replace_unknown_words_with_UNK(val_set, word_to_wid)
        test_set = self.replace_unknown_words_with_UNK(test_set, word_to_wid)

        train_set = self.add_question_ids(train_set, word_to_wid)
        val_set = self.add_question_ids(val_set, word_to_wid)
        test_set = self.add_question_ids(test_set, word_to_wid)

        train_set = self.train_encode_answers(train_set, ans_to_aid)
        val_set = self.val_encode_answers(val_set, ans_to_aid)

        print('Saving processed datasets...')

        if not os.path.exists(self.processed_dir):
            subprocess.run(['mkdir', '-p'] + [self.processed_dir])

        torch.save(train_set, os.path.join(self.processed_dir, 'train2014_processed.pth'))
        torch.save(val_set, os.path.join(self.processed_dir, 'val2014_processed.pth'))
        torch.save(test_set, os.path.join(self.processed_dir, 'test2015_processed.pth'))
        torch.save(wid_to_word, os.path.join(self.processed_dir, 'wid_to_word.pth'))
        torch.save(word_to_wid, os.path.join(self.processed_dir, 'word_to_wid.pth'))
        torch.save(ans_to_aid, os.path.join(self.processed_dir, 'ans_to_aid.pth'))
        torch.save(aid_to_ans, os.path.join(self.processed_dir, 'aid_to_ans.pth'))

        print('Finished processing annotations and questions.')















