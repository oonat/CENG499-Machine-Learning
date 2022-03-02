import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, layer_num, layer_features, act_func):
        super().__init__()

        if(act_func == "relu"):
            self.act_func = F.relu
        elif(act_func == "sigmoid"):
            self.act_func = F.sigmoid
        else:
            self.act_func = F.tanh

        self.layer_num = layer_num
        self.layers = nn.ModuleList([nn.Linear(layer_features[i], layer_features[i + 1]) for i in range(layer_num)])

    def forward(self, x):
        x = torch.flatten(x, 1)

        for i in range(self.layer_num - 1):
            x = self.act_func(self.layers[i](x))

        x = self.layers[-1](x)

        # not have to add softmax layer here
        return x