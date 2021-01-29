from .resnet import ResNet18, ResNet34, ResNet50
from .matchers import MeanPool
import torch.nn as nn

class FullModelWrapper(nn.Module):
    def __init__(self, img_encoder, snd_encoder, matcher, loss_function):
        super().__init__()
        self.img_encoder   = img_encoder
        self.snd_encoder   = snd_encoder
        self.matcher       = matcher
        self.loss_function = loss_function

