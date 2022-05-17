import time
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
from configs.global_config import SHOW_INFO_MESSAGES


def prompt_yes_no_msg(msg):
    print(f"{msg}(Y/N): ", end='')
    ans = input().lower()
    if ans == 'y':
        return True
    else:
        return False


def cprint(msg, confirmed=True):
    if confirmed:
        print(msg)


def info(msg, with_time=True, verbose=True):
    """
    Show info message
    """
    if SHOW_INFO_MESSAGES == True and verbose == True:
        if with_time == True:
            print(f'[{datetime.now()}] -- {msg}')
        else:
            print(msg)
    # time.sleep(0.5) # to force buffer dump to stdout (used for tqdm call)

def blank_line():
    print()

def get_epoch_losses(metrics_file, n_epochs: int):
    metrics = torch.load(metrics_file)
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']

    tr_bs = len(train_losses) // n_epochs
    val_bs = len(val_losses) // n_epochs

    tr_ep_losses = [np.mean(train_losses[tr_bs * i:tr_bs * (i + 1)]) for i in range(n_epochs)]
    val_ep_losses = [np.mean(val_losses[val_bs * i:val_bs * (i + 1)]) for i in range(n_epochs)]
    return {'train_losses': tr_ep_losses, 'val_losses': val_ep_losses}
    
def plot_losses(plot_name, train_loss, val_loss, mark_min=True, save_path=None):
    x = [i for i in range(1, len(train_loss) + 1)]
    plt.plot(x, train_loss, color='red', linestyle='dashed', linewidth=1, markersize=12, label='Train')
    plt.plot(x, val_loss, color='green', linestyle='dashed', linewidth=1, markersize=12, label='Val')
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend(fontsize=14)
    plt.title(f"{plot_name} -- loss = {min(val_loss):.4f}")

    if mark_min:  # TODO: fix setting x_text and y_text
        min_val_loss = min(val_loss)
        x_point = val_loss.index(min_val_loss) + 1
        y_point = min_val_loss
        x_text = 0.90 * x_point
        y_text = 6.0
        plt.annotate(f"loss = {min_val_loss:0.5f}", xy=(x_point, y_point), xytext=(x_text, y_text),
                     arrowprops=dict(arrowstyle='Fancy', facecolor='black'))

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

