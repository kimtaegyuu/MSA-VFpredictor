import esm
import torch
import itertools
import os
import numpy as np
from torch import nn
import torch as th
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, batch_size = 1):
        super(net, self).__init__()     
        self.ffc1 = nn.Linear(1024*768+2, 4096)
        self.ffc2 = nn.Linear(4096, 1024)
        self.ffc3 = nn.Linear(1024, 512)
        self.ffc4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x,blast):
        x = th.flatten(x,start_dim=1)
        x = torch.cat([x,blast],dim=1)
        x = F.leaky_relu(self.ffc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.ffc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.ffc3(x))
        x = self.dropout(x)
        x = F.softmax(self.ffc4(x),dim=1)
        return x
