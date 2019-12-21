import numpy as np
class LR_List_Scheduler:
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
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.epoch_to_lr[epoch]# -*- coding: utf-8 -*-

