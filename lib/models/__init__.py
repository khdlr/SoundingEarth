from .resnet import ResNet18, ResNet34, ResNet50
from .resnet1d import ResNet1D18, ResNet1D34, ResNet1D50
from .reducers import *
import torch.nn as nn

class FullModelWrapper(nn.Module):
    def __init__(self, img_encoder, snd_encoder, loss_function):
        super().__init__()
        self.img_encoder   = img_encoder
        self.snd_encoder   = snd_encoder
        self.loss_function = loss_function

