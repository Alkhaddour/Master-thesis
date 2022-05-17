from typing import List
import torch
import numpy as np

from models.performer import PerformerSED
from utils.display_utils import get_epoch_losses, plot_losses


# def get_epoch_losses(metrics_file, n_epochs: int):
#     metrics = torch.load(metrics_file)
#     train_losses = metrics['train_losses']
#     val_losses = metrics['val_losses']

#     tr_bs = len(train_losses) // n_epochs
#     val_bs = len(val_losses) // n_epochs

#     tr_ep_losses = [np.mean(train_losses[tr_bs * i:tr_bs * (i + 1)]) for i in range(n_epochs)]
#     val_ep_losses = [np.mean(val_losses[val_bs * i:val_bs * (i + 1)]) for i in range(n_epochs)]
#     return {'train_losses': tr_ep_losses, 'val_losses': val_ep_losses}


file_ = "./outputs/Audioset--clustering_pretrain--V1.4/BZ-Audioset-clustering_pretrain--Epoch16--TMP--metrics.pkl"
losses = get_epoch_losses(file_, 16)

plot_losses("Losses", losses['train_losses'], losses['val_losses'], mark_min=False,
            save_path='./outputs/Audioset--clustering_pretrain--V1.4/losses.png')

