import torch
from torch import nn, optim
import torch.nn.functional as F
from data_setup import config
from arch_transformer import T_Block


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, num_layers, act):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dmodel)
        self.pos_emb = nn.Embedding(config.seq_len, dmodel)
        self.blocks = nn.Sequential(*[T_Block(dmodel, num_heads, act) for _ in range(num_layers)]) # unpacks the list and passes them as arguments
        self.ln = nn.LayerNorm(dmodel)
        self.fc = nn.Linear(dmodel, vocab_size)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss