import torch
import numpy as np
import matplotlib.pyplot as plt
from config import cfg, state
import wandb

def _rgb(image):
    return np.clip(image[[3, 2, 1]].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)

@torch.no_grad()
def log_image(image, timestamp, mask, reconst, tag):
    T = image.shape[0]
    ROWS = 2

    placeholder = np.ones_like(_rgb(image[0]))
    row1 = np.concatenate([_rgb(i) for i in image], axis=1)
    row2 = []

    j = 0
    for i in range(T):
        if mask[i]:
            row2.append(placeholder)
        else:
            row2.append(_rgb(reconst[j]))
            j += 1

    row2 = np.concatenate(row2, axis=1)
    ary = np.concatenate([row1, row2], axis=0)
    wandb.log({tag: wandb.Image(ary)}, step=state.BoardIdx)
