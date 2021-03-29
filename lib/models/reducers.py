import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class ImageMeanPool(nn.Module):
    def forward(self, Z_img):
        # Calculate mean over Z_img
        Z_img = Z_img.reshape(*Z_img.shape[:2], -1)
        Z_img = Z_img.mean(dim=2)

        return Z_img


class SoundMeanPool(nn.Module):
    def forward(self, Z_snd, splits):
        # Calculate mean over Z_snd
        Z_snd_splits = torch.split(Z_snd, splits, dim=0)
        Z_snd = []
        for z_snd in Z_snd_splits:
            Z_snd.append(torch.mean(z_snd, dim=0))
        Z_snd = torch.stack(Z_snd)

        return Z_snd


class RandomAudioSample(nn.Module):
    def forward(self, Z_snd, splits):
        Z_snd_splits = torch.split(Z_snd, splits, dim=0)
        Z_snd = []
        for z_snd in Z_snd_splits:
            Z_snd.append(z_snd[torch.randint(0, z_snd.shape[0], [])])
        Z_snd = torch.stack(Z_snd)

        return Z_snd


class AudioAttention(nn.Module):
    def __init__(self):
        raise NotImplementedError("Not yet adapted for Reducers framework!")

    def forward(self, Z_img, Z_snd, snd_splits, attn_dims=16):
        # Calculate mean over Z_img
        C = Z_img.shape[1]
        A = attn_dims
        Z_img = Z_img.reshape(*Z_img.shape[:2], -1)
        Z_img = Z_img.mean(dim=2)
        Z_img, query = Z_img.split([C - A, A], dim=1)

        # Calculate mean over Z_snd
        Z_snd_splits = torch.split(Z_snd, snd_splits, dim=0)
        Z_snd = []
        for z_snd in Z_snd_splits:
            if self.training:
                selection = torch.randperm(z_snd.shape[0])[:5]
                z_snd = z_snd[selection]
            value, key = z_snd.split([C - A, A], dim=1)
            attn = torch.einsum('ba,sa->bs', query, key)
            attn = F.softmax(attn, dim=1)
            Z_snd.append(torch.einsum('bs,sc->bc', attn, value))
        Z_snd = torch.stack(Z_snd, dim=0)

        M_img = Z_img.unsqueeze(0).expand(Z_snd.shape[0], Z_img.shape[0], Z_img.shape[1])
        M_snd = Z_snd

        return M_img, M_snd
