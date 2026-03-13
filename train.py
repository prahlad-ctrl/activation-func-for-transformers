import torch
from torch import nn, optim
import torch.nn.functional as F
from data_setup import config, get_batch, vocab_size
from LangModel import LanguageModel


@torch.no_grad()
def est_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for i in range(200):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        
        out[f'{split}_loss'] = losses.mean()
    model.train()
    
    return out

# setup loop to test all act funcs
act_funcs = ['relu', 'gelu', 'swish', 'swiglu']
results = {}
for act in act_funcs:
    print(f"training with: {act}")
    val_losses = []
    model = LanguageModel(vocab_size, config.dmodel, num_heads=4, num_layers=2, act=act).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    
    for epoch in range(config.num_epochs):
        if epoch % 300 == 0:
            e_loss = est_loss(model)
            val_losses.append(e_loss['val_loss'])
            print(f"epoch: {epoch}, training_loss: {e_loss['train_loss']:.4f}, val_loss: {e_loss['val_loss']:.4f}")
        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    results[act] = val_losses
    
# plot for all act funcs as ref
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.title("Validation Loss for Different Activation Functions")
plt.xlabel("Epochs")
plt.ylabel("Loss")
for act, losses in results.items():
    x_steps = [i*2 for i in range(len(losses))]
    plt.plot(x_steps, losses, label=act)
    
plt.legend()
plt.grid()

plt.savefig("activation_comparison.png")
plt.show()