import torch
from torch import nn, optim
import torch.nn.functional as F


class CustomFFN(nn.Module):
    def __init__(self, dmodel, act):
        super().__init__()
        self.act = act
        if act == 'swiglu':
            hidden_dim = int(8* dmodel/3) # param count matches the standard FFN
            self.w1 = nn.Linear(dmodel, hidden_dim)
            self.w2 = nn.Linear(dmodel, hidden_dim)
            self.w3 = nn.Linear(hidden_dim, dmodel)
        else:
            hidden_dim = 4* dmodel
            self.fc1 = nn.Linear(dmodel, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dmodel)
    
    def forward(self, x):
        if self.act == 'relu':
            return self.fc2(F.relu(self.fc1(x)))
        elif self.act == 'gelu':
            return self.fc2(F.gelu(self.fc1(x)))
        elif self.act == 'swish':
            return self.fc2(F.silu(self.fc1(x)))
        elif self.act == 'swiglu':
            gate = F.silu(self.w1(x))
            value = self.w2(x)
            hidden = gate*value
            
        return self.w3(hidden)