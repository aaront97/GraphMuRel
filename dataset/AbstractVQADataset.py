import torch
from torch.utils.data import Dataset
import os 
import progressbar
import json
from collections import Counter

class AbstractVQADataset(Dataset):
	def __init__(self, 
		processed_dir='processed_splits', 
		model='baseline',
		root_dir='D:/VQA/',
		no_answers=1000,
		sample_answers=True):

		self.root_dir = root_dir
		self.processed_dir = processed_dir
		self.splits = ['train2014', 'val2014' , 'train2015']
		if not os.path.exists(processed_dir):
			os.system('mkdir -p' + self.processed_dir)
			self.process(model)
		self.model = baseline

	def tokenize(self, s):
		t_str = s.lower()
	    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
	        t_str = re.sub( i, '', t_str)
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
		return answer_vocabulary

	def get_top_question_words()


	def add_most_frequent_answer(self, split_set):
		for annotation in split_set['annotations']:
			annotation['most_frequent_answer'] = annotation['multiple_choice_answer']
		return split_set

	def add_images(self, split_set):
		for question in split_set['questions']:
			question['image_name'] = 'COCO_%s_%012d.jpg'%(split_set['data_subtype'],question['image_id'])
		return split_set


	def process(self):
		for split in self.splits:
			split_dir = os.path.join(self.processed_dir, split)
			if not os.path.exists(split_dir):
				os.system('mkdir -p' + split_dir)

		path_train_anno = os.path.join(self.root_dir, 'Annotations', 'v2_mscoco_train2014_annotations')
		path_val_anno = os.path.join(self.root_dir, 'Annotations', 'v2_mscoco_val2014_annotations')
		path_train_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_train2014_questions')
		path_val_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_val2014_questions')
		path_test_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_test2015_questions')

		train_set = self.merge_questions_with_annotations(json.load(path_train_ques),
			json.load(path_train_anno))
		val_set = self.merge_questions_with_annotations(json.load(path_val_ques),
			json.load(path_val_anno))
		test_set = json.load(path_test_ques)

		train_set = self.add_images(train_set)
		val_set = self.add_images(val_set)
		test_set = self.add_images(test_set)

		train_set = self.add_most_frequent_answer(train_set)
		val_set = self.add_most_frequent_answer(val_set)
		test_set = self.add_most_frequent_answer(test_set)

		#We only deal with the top #no_answers
		aid_to_ans = self.get_top_answers(train_set)
		ans_to_aid = {word: index for index, word in enumerate(aid_to_ans)}








