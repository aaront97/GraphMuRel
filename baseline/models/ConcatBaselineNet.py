import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatBaselineNet(nn.Module):
    #Todo Dropout?
    def __init__(self, input_dim, out_dim, hidden_list, dropout):
        #Hidden list expects a list of arguments that specifies
        #The depth and length of the hidden layers, e.g.
        # [4448, 1024, 512, 3000]
        super(ConcatBaselineNet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden = nn.ModuleList()
        self.dropout = dropout
#         self.layernorm1 = nn.LayerNorm(input_dim)
#         self.fc1 = nn.Linear(input_dim, out_dim)
        if not hidden_list or len(hidden_list) <= 2:
            print("Initialising MLP with one hidden layer")
            self.hidden.append(nn.LayerNorm(input_dim))
            self.hidden.append(nn.Linear(input_dim, out_dim))
        else:
            assert hidden_list[0] == input_dim and hidden_list[-1] == out_dim, \
             "Please make the first and last element equal to input_dim and out_dim respectively"
            for i in range(len(hidden_list) - 2):
                self.hidden.append(nn.LayerNorm(hidden_list[i]))
                self.hidden.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
                self.hidden.append(nn.Dropout(p=self.dropout))
            self.hidden.append(nn.LayerNorm(hidden_list[-2]))
            self.hidden.append(nn.Linear(hidden_list[-2], hidden_list[-1]))
                

    def forward(self, x):
#         x = self.layernorm1(x)
#         x = self.fc1(x)
#         return x

        for i in range(len(self.hidden) - 1):
            layer = self.hidden[i]
            if isinstance(layer, nn.LayerNorm) or isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        #No activation at last layer
        x = self.hidden[-1](x)
        return x

        