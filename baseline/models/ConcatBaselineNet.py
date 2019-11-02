import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatBaselineNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_list):
        #Hidden list expects a list of arguments that specifies
        #The depth and length of the hidden layers, e.g.
        # [4448, 1024, 512, 3000]
        super(ConcatBaselineNet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden = nn.ModuleList()
        if not hidden_list or len(hidden_list) == 1:
            print("Hidden List is empty! Initialising MLP with one hidden layer")
            self.hidden.append(nn.Linear(input_dim, out_dim))
        else:
            assert hidden_list[0] == input_dim and hidden_list[-1] == out_dim, \
             "Please make the first and last element equal to input_dim and out_dim respectively"
            for i in range(len(hidden_list) - 1):
                self.hidden.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
    
    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return x
        
        