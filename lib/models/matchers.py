import torch
import torch.nn as nn


class MeanPool(nn.Module):
    def forward(self, Z_img, Z_snd, snd_splits):
        # Calculate mean over Z_img
        Z_img = Z_img.reshape(*Z_img.shape[:2], -1)
        Z_img = Z_img.mean(dim=2)

        # Calculate mean over Z_snd
        Z_snd_splits = torch.split(Z_snd, snd_splits, dim=0)
        Z_snd = []
        for z_snd in Z_snd_splits:
            Z_snd.append(torch.mean(z_snd, dim=0))
        Z_snd = torch.stack(Z_snd)

        M_img = Z_img.unsqueeze(0).expand(Z_snd.shape[0], Z_img.shape[0], Z_img.shape[1])
        M_snd = Z_snd.unsqueeze(1).expand(Z_snd.shape[0], Z_img.shape[0], Z_snd.shape[1])

        return M_img, M_snd
