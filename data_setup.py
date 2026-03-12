import torch

# using the tiny shakespeare dataset as corpus
class Config:
    batch_size = 64
    seq_len = 64
    dmodel = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    num_epochs = 3000
    
config = Config()

with open('corpus.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)

train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# to feeed model the data in chunks
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data_split)-config.seq_len, (config.batch_size,))
    x = torch.stack([data_split[i:i+config.seq_len] for i in idx]) # shape (batch_size,seq_len)
    y = torch.stack([data_split[i+1:i+config.seq_len+1] for i in idx])
    return x.to(config.device), y.to(config.device)