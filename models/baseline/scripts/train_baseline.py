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
import tqdm

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

def val_evaluate(model, epoch, val_loader, writer, criterion, size):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    count = 1
    evaluator_criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data in val_loader:
#             if count % (len(val_loader) // 4) == 0:
#                 break
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += evaluator_criterion(outputs, labels).item()
            count += 1
    avg_accuracy = correct / total
    avg_cross_entropy = running_loss / total
    print("Depth {}: Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(size, epoch, avg_accuracy, avg_cross_entropy))
    writer.add_scalar("{}/validation/avg_loss".format(size), avg_cross_entropy, epoch)
    writer.add_scalar("{}/validation/avg_accuracy".format(size), avg_accuracy, epoch)
    return avg_accuracy

def train_evaluate(model, epoch, train_loader, writer, criterion, size):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    count = 1
    evaluator_criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data in train_loader:
#             if count % (len(train_loader) // 5) == 0:
#                 break
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += evaluator_criterion(outputs, labels).item()
            count += 1
    avg_accuracy = correct / total
    avg_cross_entropy = running_loss / total
    print("Depth {}: Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(size, epoch, avg_accuracy, avg_cross_entropy))
    writer.add_scalar("{}/training/avg_loss".format(size), avg_cross_entropy, epoch)
    writer.add_scalar("{}/training/avg_accuracy".format(size), avg_accuracy, epoch)
    return avg_accuracy


def checkpoint(model, epoch, isBest, config, model_dir, writer_dir_name):
    if not os.path.exists(model_dir):
        suprocess.run(["mkdir", "-p", model_dir])
    if epoch % config['checkpoint_every'] == 0:
        model_name = writer_dir_name + "_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    if isBest:
        model_name = writer_dir_name + "_BEST.pth"
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    
def load_checkpoint_if_available(model, model_dir, writer_dir_name, load_last_epoch=True):
    best_name = os.path.join(model_dir, writer_dir_name) + "_BEST.pth"
    epoch_list = [f for f in os.listdir(model_dir) if 'epoch' in f and f.startswith(writer_dir_name)]
    epoch_list = sorted(epoch_list)
    epoch_no = 0
    if epoch_list and load_last_epoch:
        last_epoch_name = epoch_list[-1]
        epoch_no = int(last_epoch_name.split("epoch_")[1].strip('.pth'))
        model.load_state_dict(torch.load(os.path.join(model_dir, last_epoch_name)))
        return epoch_no + 1, model
    if os.path.exists(best_name) and not load_last_epoch:
        model.load_state_dict(torch.load(best_name))
        return epoch_no + 1, model
    else:
        return epoch_no + 1, model

def run():
    with open('baseline.yaml') as f:
        config = yaml.load(f)
    config = config['baseline_options']
    ROOT_DIR = '/auto/homes/bat34/VQA_PartII/baseline/'
    writer_dir_name = get_option_directory(config, ["dropout", "batch_size", "lr", "weight_decay"])
    logdir = os.path.join(ROOT_DIR, "logs", writer_dir_name)
    model_dir = os.path.join(ROOT_DIR, 'trained_models')
    
    
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
                              collate_fn=collate_fn,\
                             num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, shuffle=True, \
                            batch_size=config['batch_size'], \
                            collate_fn=collate_fn,\
                            num_workers=config['num_workers'])
    input_dim = list(train_dataset[0]['concat_vector'].size())[0]
    out_dim = len(train_dataset.ans_to_aid)
    size = config['max_depth']
    hidden_list = get_hidden_layer_list(input_dim, out_dim, size)
    model = ConcatBaselineNet(input_dim, out_dim, \
                                      hidden_list, \
                                      dropout=config['dropout'])
    start_epoch, model = load_checkpoint_if_available(model, model_dir, writer_dir_name, load_last_epoch=config['load_last_epoch'])
    print('Starting from EPOCH {}'.format(start_epoch))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],\
                                     weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    model.cuda()
    
    max_accuracy = -1
    global_iteration = 0
    for epoch in tqdm.tqdm(range(start_epoch, config['epochs'] + 1)):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(train_loader)
        local_iteration = 0
        for data in pbar:
            global_iteration += 1
            local_iteration += 1
            
            pbar.set_description("Epoch[{}] Iteration[{}/{}]".format(epoch, local_iteration, len(train_loader)))
            
            inputs, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if local_iteration % 50 == 0:
                running_loss = running_loss / 50
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(epoch, local_iteration, len(train_loader), running_loss))
                writer.add_scalar("{}/training/loss".format(size), running_loss, global_iteration )
                running_loss = 0.0

        
        #At the end of every epoch, run it on the validation and training dataset
        train_evaluate(model, epoch, train_loader, writer, criterion, size)
        accuracy = val_evaluate(model, epoch, val_loader, writer, criterion, size)
        
        isBest = False
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            isBest = True
        checkpoint(model, epoch, isBest, config, model_dir, writer_dir_name)

if __name__ == '__main__':
    run()