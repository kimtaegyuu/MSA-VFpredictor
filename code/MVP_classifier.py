from torch import nn
import torch as th
import torch.nn.functional as F
import torch

class MVP_classifier(nn.Module):
    def __init__(self, batch_size = 1):
        super(MVP_classifier, self).__init__()
        self.batch_size = batch_size
        self.fc_mat = nn.Linear(1089, 1)
        self.ffc1 = nn.Linear(20*768+2, 4096)
        self.ffc2 = nn.Linear(4096, 1024)
        self.ffc3 = nn.Linear(1024, 512)
        self.ffc4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.4)
    def forward(self, msa_composition,seqsim):
        x = th.flatten(msa_composition,start_dim=1)
        x = torch.cat([x,seqsim],dim=1)
        x = F.leaky_relu(self.ffc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.ffc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.ffc3(x))
        x = self.dropout(x)     
        x = F.softmax(self.ffc4(x),dim=1)
        return x
