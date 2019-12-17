import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import yaml
from dataset.MurelNetDataset import MurelNetDataset
from murel.models.MurelNet import MurelNet
from tensorboardX import SummaryWriter
import tqdm
import subprocess
from collections import OrderedDict
import numpy as np


def create_summary_writer(model, loader, logdir):
    batch = next(iter(loader))
    writer = SummaryWriter(logdir=logdir)
    try:
        writer.add_graph(model, batch)
    except Exception as e:
        print("Writer can't save model at {}".format(logdir))
        print(e)
    return writer

def get_option_directory(config, chosen_keys, model_name='murel'):
    res = model_name
    for key in chosen_keys:
        res += "_{}_{}".format(key, config[key])
    return res

def val_evaluate(model, epoch, val_loader, writer, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    count = 1
    evaluator_criterion = nn.NLLLoss(reduction='sum')
    with torch.no_grad():
        for data in val_loader:
            item = {\
                    'question_embedding': data['question_embedding'].cuda(), \
                    'object_features_list': data['object_features_list'].cuda(), \
                    'bounding_boxes': data['bounding_boxes'].cuda(), \
                    'answer_id': torch.squeeze(data['answer_id']).cuda()
            }
            inputs, labels = item, item['answer_id']
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += evaluator_criterion(outputs, labels).item()
            count += 1
    avg_accuracy = correct / total
    avg_cross_entropy = running_loss / total
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(epoch, avg_accuracy, avg_cross_entropy))
    writer.add_scalar("validation/avg_loss", avg_cross_entropy, epoch)
    writer.add_scalar("validation/avg_accuracy", avg_accuracy, epoch)
    return avg_accuracy

def train_evaluate(model, epoch, train_loader, writer, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    count = 1
    evaluator_criterion = nn.NLLLoss(reduction='sum')
    with torch.no_grad():
        for data in train_loader:
#             if count % (len(train_loader) // 5) == 0:
#                 break
            item = {\
                    'question_embedding': data['question_embedding'].cuda(), \
                    'object_features_list': data['object_features_list'].cuda(), \
                    'bounding_boxes': data['bounding_boxes'].cuda(), \
                    'answer_id': torch.squeeze(data['answer_id']).cuda()
            }
            inputs, labels = item, item['answer_id']
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += evaluator_criterion(outputs, labels).item()
            count += 1
    avg_accuracy = correct / total
    avg_cross_entropy = running_loss / total
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(epoch, avg_accuracy, avg_cross_entropy))
    writer.add_scalar("training/avg_loss", avg_cross_entropy, epoch)
    writer.add_scalar("training/avg_accuracy", avg_accuracy, epoch)
    return avg_accuracy


def checkpoint(model, epoch, isBest, config, model_dir, writer_dir_name, accuracy):
    if not os.path.exists(model_dir):
        subprocess.run(["mkdir", "-p", model_dir])
    writer_dir_name += '_accuracy_{}'.format(accuracy)
    if (epoch + 1) % config['checkpoint_every'] == 0:
        model_name = writer_dir_name + "_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    if isBest:
        model_name = writer_dir_name + "_epoch_{}".format(epoch)
        model_name = model_name + "_BEST.pth"
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

def generate_state_dict_compatible_with_data_parallel(old_state_dict):
    print('Generating State Dict compatible with nn.DataParallel by removing leading "module." from keys')
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        if key.startswith('module'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def add_module_in_front_of_keys(old_state_dict):
    print('Generating State Dict compatible with nn.DataParallel by adding leading "module." from keys')
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        if not key.startswith('module'):
            new_state_dict['module.' + key] = value 
        else:
            new_state_dict[key] = value
    return new_state_dict

#TODO: FIX
def load_checkpoint_if_available(model, model_dir, writer_dir_name, load_last_epoch=True, dataParallel=True):
    best_name = os.path.join(model_dir, writer_dir_name) + "_BEST.pth"
    epoch_list = [f for f in os.listdir(model_dir) if 'epoch' in f and f.startswith(writer_dir_name)]
    epoch_list = sorted(epoch_list)
    epoch_no = 0
    if epoch_list and load_last_epoch:
        last_epoch_name = epoch_list[-1]
        epoch_no = int(last_epoch_name.split("epoch_")[1].strip('.pth').strip('_BEST'))
        model_dict = torch.load(os.path.join(model_dir, last_epoch_name))
        if dataParallel:
            model_dict = add_module_in_front_of_keys(model_dict)
            model.load_state_dict(model_dict)
        else:
            model_dict = generate_state_dict_compatible_with_data_parallel(model_dict)
            model.load_state_dict(model_dict)
        return epoch_no + 1, model
    if os.path.exists(best_name) and not load_last_epoch:
        epoch_no = int(best_name.split("epoch_")[1].strip('.pth'))
        model_dict = torch.load(best_name)
        model.load_state_dict(model_dict)
        if dataParallel:
            model_dict = add_module_in_front_of_keys(model_dict)
            model.load_state_dict(model_dict)
        else:
            model_dict = generate_state_dict_compatible_with_data_parallel(model_dict)
            model.load_state_dict(model_dict)
        return epoch_no + 1, model
    else:
        return 0, model


class LR_Scheduler:
    def __init__(self, config):
        gradual_warmup_steps = config['gradual_warmup_steps']
        lr_decay_epochs = config['lr_decay_epochs']
        lr_decay_rate = config['lr_decay_rate']
        base_lr = config['lr']
        self.lr_decay_rate = lr_decay_rate
        self.base_lr = base_lr
        self.epoch_to_lr = list(np.linspace(gradual_warmup_steps[0], \
                                              gradual_warmup_steps[1], \
                                              int(gradual_warmup_steps[2])))
        self.epoch_to_lr = [base_lr * weight for weight in self.epoch_to_lr]
        max_epochs = config['epochs']
        no_warmup_steps = len(self.epoch_to_lr)
        lr_decay_epochs = set(list(range(lr_decay_epochs[0], \
                                          lr_decay_epochs[1], \
                                          lr_decay_epochs[2])))
        for i in range(no_warmup_steps, max_epochs):
            if i in lr_decay_epochs:
                self.epoch_to_lr.append(self.epoch_to_lr[-1] * self.lr_decay_rate)
            else:
                self.epoch_to_lr.append(self.epoch_to_lr[-1])
        assert len(self.epoch_to_lr) == max_epochs
    
    def update_lr(self, optimizer, epoch):
        optimizer.param_groups[0]['lr'] = self.epoch_to_lr[epoch]
#         prev_lr = optimizer.param_groups[0]['lr']
        
#         if epoch < len(self.gradual_warmup_steps):
#             optimizer.param_groups[0]['lr'] = self.gradual_warmup_steps[epoch]
#         elif epoch in self.lr_decay_epochs:
#             optimizer.param_groups[0]['lr'] = prev_lr * self.lr_decay_rate
#         else:
#             optimizer.param_groups[0]['lr'] = prev_lr

def run():
    with open('murel.yaml') as f:
        config = yaml.load(f)
    config = config['murel_options']
    ROOT_DIR = config['ROOT_DIR']
    writer_dir_name = get_option_directory(config, ['txt_enc', 'batch_size', 'lr', 'lr_decay_rate', \
                                                    'unroll_steps', 'fusion_type'])
    logdir = os.path.join(ROOT_DIR, "logs", writer_dir_name)
    model_dir = os.path.join(ROOT_DIR, 'trained_models')
    
    
    writer = SummaryWriter(logdir=logdir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('CUDA AVAILABILITY: {}, Device used: {}'.format(torch.cuda.is_available(), device))
    

    
    train_dataset = MurelNetDataset(split="train", \
                                    txt_enc=config['txt_enc'], \
                                    bottom_up_features_dir=config['bottom_up_features_dir'], \
                                    skipthoughts_dir=config['skipthoughts_dir'], \
                                    processed_dir=config['processed_dir'], \
                                    ROOT_DIR=ROOT_DIR, \
                                    vqa_dir=config['vqa_dir'])
    val_dataset =   MurelNetDataset(split="val", \
                                    txt_enc=config['txt_enc'], \
                                    bottom_up_features_dir=config['bottom_up_features_dir'], \
                                    skipthoughts_dir=config['skipthoughts_dir'], \
                                    processed_dir=config['processed_dir'], \
                                    ROOT_DIR=ROOT_DIR, \
                                    vqa_dir=config['vqa_dir'])
    # Applying Catalina's trick, reducing batch size and memory while keeping gradient updates more or less the same
    # Trick applied at epoch 17, reducing batch size to batch size // 3 and calling optimizer.step() every 3 iterations
    # and calling loss.backward() every iteration
    # https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
    
    reduction_factor = config['reduction_factor']
    batch_size = config['batch_size'] // reduction_factor
    
    train_loader = DataLoader(train_dataset, shuffle=True, \
                              batch_size=batch_size, \
                             num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, shuffle=True, \
                            batch_size=batch_size, \
                            num_workers=config['num_workers'])
    model = MurelNet(config)
    model.cuda()

    if torch.cuda.device_count() > 1:
        print('Initialising DataParallel with more than one device')
        model = nn.DataParallel(model, device_ids=[1, 2])
        start_epoch, model = load_checkpoint_if_available(model, model_dir, writer_dir_name, load_last_epoch=config['load_last_epoch'], dataParallel=True)
    else:
        start_epoch, model = load_checkpoint_if_available(model, model_dir, writer_dir_name, load_last_epoch=config['load_last_epoch'], dataParallel=False)
    print('Model loaded, all keys matched successfully')
    
    print('Starting from EPOCH {}'.format(start_epoch))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    lr_scheduler = LR_Scheduler(config)
    criterion = nn.NLLLoss()

    max_accuracy = -1
    global_iteration = 0
    for epoch in tqdm.tqdm(range(start_epoch, config['epochs'])):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(train_loader)
        local_iteration = 0
        lr_scheduler.update_lr(optimizer, epoch)
        optimizer.zero_grad()
        for data in pbar:
            global_iteration += 1
            local_iteration += 1
            pbar.set_description("Epoch[{}] Iteration[{}/{}]".format(epoch, \
                                 local_iteration, len(train_loader)))
            item = {\
                    'question_ids': data['question_ids'].cuda(), \
                    'object_features_list': data['object_features_list'].cuda(), \
                    'bounding_boxes': data['bounding_boxes'].cuda(), \
                    'answer_id': torch.squeeze(data['answer_id']).cuda()
            }

#             optimizer.zero_grad()
#             inputs, labels = item, item['answer_id']
#             outputs = model(inputs)
#             loss = criterion(outputs,labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
                
            inputs, labels = item, item['answer_id']
            outputs = model(inputs)
            loss = criterion(outputs, labels) / reduction_factor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            if local_iteration % reduction_factor == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * reduction_factor



            if local_iteration % (config['log_every'] * reduction_factor) == 0:
                running_loss = running_loss / (config['log_every'] * reduction_factor)
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(epoch, \
                      local_iteration, len(train_loader), running_loss))
                writer.add_scalar("training/loss", running_loss, global_iteration )
                running_loss = 0.0
                
        #At the end of every epoch, run it on the validation and training dataset
        train_evaluate(model, epoch, train_loader, writer, criterion)
        accuracy = val_evaluate(model, epoch, val_loader, writer, criterion)
        
        isBest = False
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            isBest = True
        checkpoint(model, epoch, isBest, config, model_dir, writer_dir_name, accuracy)

if __name__ == '__main__':
    run()
