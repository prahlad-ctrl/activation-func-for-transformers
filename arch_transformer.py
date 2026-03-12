import torch
from torch import nn, optim
import torch.nn.functional as F
from arch_ffn import CustomFFN
from data_setup import config


class Head(nn.Module):
    def __init__(self, dmodel, head_size):
        super().__init__()
        self.key = nn.Linear(dmodel, head_size, bias=False)
        self.query = nn.Linear(dmodel, head_size, bias=False)
        self.value = nn.Linear(dmodel, head_size, bias=False)
        
        # mask to prevent attention to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(config.seq_len, config.seq_len)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1)* C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # softmax makes -inf to 0
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        v = self.value(x)
        out = weights @ v
        
        return out # (B,T,head_size)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dmodel):
        super().__init__()
        self.heads = nn.ModuleList([Head(dmodel, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, dmodel)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        
        return out
    

class T_Block(nn.Module):
    def __init__(self, dmodel, num_heads, act):
        super().__init__()
        head_size = dmodel//num_heads
        self.attn = MultiHeadAttention(num_heads, head_size, dmodel)
        self.ffn = CustomFFN(dmodel, act)
        self.ln1 = nn.LayerNorm(dmodel)
        self.ln2 = nn.LayerNorm(dmodel)
    
    def forward(self, x):
        x = x+ self.attn(self.ln1(x))
        x = x+ self.ffn(self.ln2(x))
        
        return x