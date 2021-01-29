import hashlib
import numpy as np
import torch.optim as opt
from . import models, loss_functions


def md5(obj):
    if type(obj) is not str:
        obj = str(obj).encode('utf8')
    return hashlib.md5(obj).hexdigest()[:16]


def get_model(model_name):
    try:
        return models.__dict__[model_name]
    except KeyError:
        raise ValueError(f'Can\'t provide Model called "{model_name}"')


def get_optimizer(name):
    try:
        return opt.__dict__[name]
    except KeyError:
        raise ValueError(f'Can\'t provide Optimizer called "{name}"')


def get_loss_function(name):
    try:
        return loss_functions.__dict__[name]
    except KeyError:
        raise ValueError(f'Can\'t provide Loss Function called "{name}"')
