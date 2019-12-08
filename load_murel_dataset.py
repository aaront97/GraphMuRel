from skipthoughts import BayesianUniSkip
from dataset.MurelNetDataset import MurelNetDataset
dataset = MurelNetDataset(bottom_up_features_dir='/media/bat34/Elements/play_murel/2018-04-27_bottom-up-attention_fixed_36/', \
                          vqa_dir='/media/bat34/Elements/VQA/', \
                          ROOT_DIR='/home/bat34/Desktop/VQA_PartII/', \
                          skipthoughts_dir='/home/bat34/Desktop/VQA_PartII/data/skipthoughts', \
                          processed_dir='/home/bat34/Desktop/VQA_PartII/data/processed_splits')
