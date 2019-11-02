from . import AbstractVQADataset
import torch
import os
from .TextEncFactory import get_text_enc
class ConcatBaselineDataset(AbstractVQADataset):
	def __init__(self, 
              preprocessed_images_dir='~/VQA/BaselineTraining',
              split='train',
              ROOT_DIR='~/VQA_PartII/',
              txt_enc='BayesianUniSkip'):
        super(ConcatBaselineDataset, self).__init__()
        self.image_features = torch.load( \
               os.path.join(preprocessed_images_dir, split, \
                'baseline_{}_cnn_features.pth'.format(split)))
        skipthoughts_dir = os.path.join(ROOT_DIR, 'data', 'skipthoughts')
        self.text_enc = get_text_enc(skipthoughts_dir, txt_enc, self.wid_to_word)
        if split == 'train':
            self.dataset = self.train_set
        if split == 'val':
            self.dataset = self.val_set
        if split == 'test':
            self.dataset = self.test_set
        
        
	def __getitem__(self, idx):
        question = self.dataset['questions'][idx]
        question_ids = torch.LongTensor(question['question_ids'])
        image_name = question['image_name']
        question_vector = self.text_enc([question_ids], [len(question_ids)])
        #Squeeze question_vector as it is in batch
        #Size 2400
        question_vector = torch.squeeze(question_vector)
        #Size 2048
        image_vector = self.image_features[image_name]
        #Size 4448
        training_vector = torch.cat((image_vector, question_vector), 0)
        answer = self.dataset['annotations'][idx]['answer_id']
        return (training_vector, answer)
        
	def __len__(self):
        return len(self.dataset['questions'])

