# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import yaml
from dataset.ConcatBaselineDataset import ConcatBaselineDataset
from baseline.models.ConcatBaselineNet import ConcatBaselineNet
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import transforms.transforms as trfm
import subprocess
#CODE NOT WORKING, GPU NOT UTILISED

def get_hidden_layer_list(input_dim, out_dim, size):
    if size <= 1:
        return [input_dim, out_dim]
    gap = input_dim - out_dim
    if gap % size == 0:
        repeat = size
    else:
        repeat = size - 1
    hidden_list = [input_dim]
    step = gap // size
    for i in range(1, repeat + 1):
        hidden_list.append(input_dim - i * step)
    if gap % size != 0:
        hidden_list.append(out_dim)
    return hidden_list

def create_summary_writer(model, loader, logdir):
    batch = next(iter(loader))
    writer = SummaryWriter(logdir=logdir)
    try:
        writer.add_graph(model, batch['concat_vector'])
    except Exception as e:
        print("Writer can't save model at {}".format(logdir))
        print(e)
    return writer

def get_option_directory(config, chosen_keys, model_name='concatbaseline'):
    res = "depth_{}_".format(config['max_depth']) + model_name
    for key in chosen_keys:
        res += "_{}_{}".format(key, config[key])
    return res

def log_training_results(engine, train_loader, evaluator, writer, size):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_cross_entropy = metrics['cross_entropy']
        print("Depth {}: Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(size, engine.state.epoch, avg_accuracy, avg_cross_entropy))
        writer.add_scalar("{}/training/avg_loss".format(size), avg_cross_entropy, engine.state.epoch)
        writer.add_scalar("{}/training/avg_accuracy".format(size), avg_accuracy, engine.state.epoch)
          
def log_iteration_results(engine, train_loader, writer, size):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % 50 == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
    writer.add_scalar("{}/training/loss".format(size), engine.state.output, engine.state.iteration)

#TODO: IMPROVE THIS, A HACK
max_accuracy = -1
def log_and_checkpoint_validation_results(engine, val_loader, evaluator, \
                                          writer, size, \
                                          checkpoint_every, model_dir, \
                                          model_name, model):
        if not os.path.exists(model_dir):
            subprocess.run(["mkdir", "-p", model_dir])
        global max_accuracy
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_cross_entropy = metrics['cross_entropy']
        print("Depth {}: Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(size, engine.state.epoch, avg_accuracy, avg_cross_entropy))
        if engine.state.epoch % checkpoint_every == 0:
            out_name = model_name + "_epoch_{}.pth".format(engine.state.epoch)
            torch.save(model.state_dict(), os.path.join(model_dir, out_name))
        if avg_accuracy > max_accuracy:
            out_name = model_name + "BEST.pth"
            torch.save(model.state_dict(), os.path.join(model_dir, out_name))
            max_accuracy = avg_accuracy
        writer.add_scalar("{}/validation/avg_loss".format(size), avg_cross_entropy, engine.state.epoch)
        writer.add_scalar("{}/validation/avg_accuracy".format(size), avg_accuracy, engine.state.epoch)
        

def run():
    with open('baseline.yaml') as f:
        config = yaml.load(f)
    config = config['baseline_options']
    ROOT_DIR = '/auto/homes/bat34/VQA_PartII/baseline/'
    option_dir_name = get_option_directory(config, ["dropout", "batch_size", "lr", "weight_decay"])
    logdir = os.path.join(ROOT_DIR, "logs", option_dir_name)
    writer = SummaryWriter(logdir=logdir)
    max_depth = config['max_depth']
    min_depth = config['min_depth']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('CUDA AVAILABILITY: {}, Device used: {}'.format(torch.cuda.is_available(), device))
    
    train_dataset = ConcatBaselineDataset(split="train", txt_enc=config['txt_enc'])
    val_dataset = ConcatBaselineDataset(split="val", txt_enc=config['txt_enc'])
    collate_fn = trfm.Compose([\
                              trfm.ConvertBatchListToDict(), \
                              trfm.CreateBatchItem(), \
                              trfm.PrepareBaselineBatch() \
            ])
    train_loader = DataLoader(train_dataset, shuffle=True, \
                              batch_size=config['batch_size'], \
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, \
                            batch_size=config['batch_size'], \
                            collate_fn=collate_fn)
    input_dim = list(train_dataset[0]['concat_vector'].size())[0]
    out_dim = len(train_dataset.ans_to_aid)
    
    
    for size in range(min_depth, max_depth + 1):
        print('Current depth: {}'.format(size))
        hidden_list = get_hidden_layer_list(input_dim, out_dim, size)
        model = ConcatBaselineNet(input_dim, out_dim, \
                                      hidden_list, \
                                      dropout=config['dropout'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],\
                                     weight_decay=config['weight_decay'])
        
        trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device='cuda')
        evaluator = create_supervised_evaluator(model,\
                                            metrics={'accuracy': Accuracy(),
                                                     'cross_entropy': Loss(F.cross_entropy)},\
                                            device='cuda')
        pbar_train = ProgressBar()
        pbar_train.attach(trainer, ['loss'])
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, train_loader, \
                                  evaluator, writer, size)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_and_checkpoint_validation_results, val_loader, \
                                  evaluator, writer, size, config['checkpoint_every'], \
                                  "/auto/homes/bat34/VQA_PartII/baseline/trained_models/", \
                                  "depth_{}_{}".format(size, option_dir_name), model)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_iteration_results, train_loader, writer, size)
        trainer.run(train_loader, max_epochs=config['epochs'])
        

if __name__ == "__main__":
    run()
# -*- coding: utf-8 -*-

