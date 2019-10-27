import torch
from torch.utils.data import Dataset
import os 
import progressbar
import json
from collections import Counter

class AbstractVQADataset(Dataset):
	def __init__(self, 
		processed_dir='data/processed_splits', 
		model='baseline',
		root_dir='D:/VQA/',
		no_answers=1000,
		sample_answers=False,
		skipthoughts_dir='data/skipthoughts'):

		#Todo: Rationalise sampling answers?
		self.skipthoughts_dir = skipthoughts_dir
		self.root_dir = root_dir
		self.processed_dir = processed_dir
		if not os.path.exists(processed_dir):
			os.system('mkdir -p' + self.processed_dir)
			self.process()

	def tokenize(self, s):
		#we don't replace # because # is used to refer to number of items
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
		print('Length of answer_vocabulary: {}, Original no. of answers: {}'.format(len(answer_vocabulary), len(sorted_counter)))
		return answer_vocabulary

	def get_skipthoughts_dictionary(self):
		dictionary_path = os.path.join(self.skipthoughts_dir, 'dictionary.txt')
		if not os.path.exists(dictionary_path):
			os.system('wget {} -P {}'.format('http://www.cs.toronto.edu/~rkiros/models/dictionary.txt', self.skipthoughts_dir))
		with open(dictionary_path, 'r') as f:
			dict_list = f.readlines()
		return set(dict_list)

	def get_known_words(self, split_set):
		#Concern ourselves with words that the pretrained embedding can embed
		skipthoughts_dictionary = self.get_skipthoughts_dictionary()
		questions_list = split_set['questions']
		known_words_list = []
		unknown_words_list = []
		for question_dict in questions_list:
			ques = question_dict['question']
			for word in ques['question_tokens']:
				if word in skipthoughts_dictionary:
					known_words_list.append(word)
				else:
					unknown_words_list.append(word)
		known_words_list.append('UNK')
		print('No. of known words: {}, No. of unknown words : {}, Percentage Loss of words: {}%' \
			.format(len(known_words), len(unknown_words), (len(unknown_words) / (len(known_words) + len(unknown_words))) * 100))
		return known_words_list, unknown_words_list

	def add_most_frequent_answer(self, split_set):
		for annotation in split_set['annotations']:
			annotation['most_frequent_answer'] = annotation['multiple_choice_answer']
		return split_set

	def add_images(self, split_set):
		for question in split_set['questions']:
			question['image_name'] = 'COCO_%s_%012d.jpg'%(split_set['data_subtype'],question['image_id'])
		return split_set

	def tokenize_questions(self, split_set):
		print('Tokenizing questions for {}'.format(splist_set['data_subtype']))
		for i in progressbar.progressbar(len(range(len(split_set['questions'])))):
			question = split_set['question'][i]
			question['question_tokens'] = self.tokenize(question['question'])
		return split_set

	def replace_unknown_words_with_UNK(self, split_set, word_to_wid):
		questions_list = split_set['questions']
		for question_dict in questions_list:
			question_dict['question_tokens_UNK'] = \
				[word if word in word_to_wid for word in question_dict['question_tokens']]
		return split_set

	def add_question_ids(self, split_set, word_to_wid):
		questions_list = split_set['questions']
		for question_dict in questions_list:
			question_ids = [word_to_wid[word] for word in question_dict['question_tokens_UNK']]
		return split_set

	def remove_question_if_not_top_answer(self, split_set, ans_to_aid):
		print('Removing questions if they have infrequent answers')
		new_annotations = []
		new_questions = []
		if(len(split_set['questions'] != len(split_set['annotations']))):
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
	

		path_train_anno = os.path.join(self.root_dir, 'Annotations', 'v2_mscoco_train2014_annotations')
		path_val_anno = os.path.join(self.root_dir, 'Annotations', 'v2_mscoco_val2014_annotations')
		path_train_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_train2014_questions')
		path_val_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_val2014_questions')
		path_test_ques = os.path.join(self.root_dir, 'Questions', 'v2_OpenEnded_mscoco_test2015_questions')

		train_set = self.merge_questions_with_annotations(json.load(path_train_ques),
			json.load(open(path_train_anno)))
		val_set = self.merge_questions_with_annotations(json.load(path_val_ques),
			json.load(open(path_val_anno)))

		#We don't have annotations for the test set
		test_set = json.load(open(path_test_ques))

		train_set = self.add_images(train_set)
		val_set = self.add_images(val_set)
		test_set = self.add_images(test_set)

		train_set = self.add_most_frequent_answer(train_set)
		val_set = self.add_most_frequent_answer(val_set)

		train_set = self.tokenize_questions(train_set)
		val_set = self.tokenize_questions(val_set)
		test_set = self.tokenize_questions(test_set)

		#We only deal with the top #no_answers
		aid_to_ans = self.get_top_answers(train_set)
		ans_to_aid = {word: index for index, word in enumerate(aid_to_ans)}

		known_words, unknown_words = self.get_known_words(train_set)
		wid_to_word = {idx: word for idx, word in enumerate(known_words)}
		word_to_wid = {word:idx} for idx, word in enumerate(known_words)}

		train_set = self.remove_question_if_not_top_answer(train_set, ans_to_aid)

		train_set = self.replace_unknown_words_with_UNK(train_set, word_to_wid)
		val_set = self.replace_unknown_words_with_UNK(val_set, word_to_wid)
		test_set = self.replace_unknown_words_with_UNK(test_set, word_to_wid)

		train_set = self.add_question_ids(train_set, word_to_wid)
		val_set = self.add_question_ids(val_set, word_to_wid)
		test_set = self.add_question_ids(test_set, word_to_wid)

		train_set = self.train_encode_answers(train_set, ans_to_aid)
		val_set = self.val_encode_answers(train_set, ans_to_aid)

		print('Saving processed datasets...')

		if not os.path.exists(self.processed_dir):
			os.system('mkdir -p ' + self.processed_dir)

		torch.save(train_set, os.path.join(self.processed_dir, 'train2014_processed.pth'))
		torch.save(val_set, os.path.join(self.processed_dir, 'val2014_processed.pth'))
		torch.save(test_set, os.path.join(self.processed_dir, 'test2015_processed.pth'))

		print('Finished processing annotations and questions.')















