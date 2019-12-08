from dataset.AbstractVQADataset import AbstractVQADataset
import torch
import os
from dataset.TextEncFactory import get_text_enc
import transforms.transforms as transforms
class ConcatBaselineDataset(AbstractVQADataset):
    def __init__(self, \
              preprocessed_images_dir='/auto/homes/bat34/VQA/BaselineTraining', \
              split='train', \
              ROOT_DIR='/auto/homes/bat34/VQA_PartII/', \
              txt_enc='BayesianUniSkip',\
              is_test_dev=True, \
              collate_fn=None, \
              processed_dir='/auto/homes/bat34/VQA_PartII/data/processed_splits', \
        	  model='murel',\
        	  vqa_dir='/auto/homes/bat34/VQA',\
        	  no_answers=3000,\
        	  sample_answers=False,\
        	  skipthoughts_dir='/auto/homes/bat34/VQA_PartII/data/skipthoughts'):
        #TODO: include skipthoughts dropout?
        super(ConcatBaselineDataset, self).__init__(
                processed_dir=processed_dir, 
                model="baseline", 
                vqa_dir=vqa_dir,
                no_answers=no_answers,
                sample_answers=sample_answers,
                skipthoughts_dir=skipthoughts_dir)
        
        self.is_test_dev = is_test_dev
        self.collate_fn = transforms.Compose([ \
                transforms.ConvertBatchListToDict(), \
                transforms.CreateBatchItem() \
                ]) if collate_fn is None else collate_fn
        self.image_features = torch.load( \
               os.path.join(preprocessed_images_dir, split, \
                'baseline_{}_cnn_features.pth'.format(split)))
        
        self.split = split
        skipthoughts_dir = os.path.join(ROOT_DIR, 'data', 'skipthoughts')
        self.text_enc = get_text_enc(skipthoughts_dir, txt_enc, [word for key, word in self.wid_to_word.items()])

    def __getitem__(self, idx):
        item = {}
        question = self.dataset['questions'][idx]
        item['index'] = idx
        item['question_id'] = question['question_id']
        item['image_name'] = question['image_name']
        question_ids = torch.LongTensor([question['question_ids']])
        image_name = question['image_name']
        question_vector = self.text_enc(question_ids, [len(question_ids)])
        #Squeeze question_vector as it is in batch
        #Size 2400
        question_vector = torch.squeeze(question_vector).detach()
        #Size 2048
        image_vector = self.image_features[image_name].detach()
        #Size 4448
        if self.split != 'test':
            annotation = self.dataset['annotations'][idx]
            item['answer_id'] = torch.LongTensor([annotation['answer_id']])
            item['answer'] = annotation['most_frequent_answer']
            item['question_type'] = annotation['question_type']
        concat_vector = torch.cat((image_vector, question_vector), 0)
        item['concat_vector'] = concat_vector
        return item
        
    def __len__(self):
        return len(self.dataset['questions'])

