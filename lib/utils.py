import hashlib
import numpy as np
import torch.optim as opt
import torch.nn as nn
try:
    import apex
except ImportError:
    pass
from . import models, loss_functions


class TupleSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, *args):
        for mod in self.module_list:
            if type(args) is tuple:
                args = mod(*args)
            else:
                args = mod(args)

        return args

    def __getitem__(self, idx):
        return self.module_list[idx]


def md5(obj):
    if type(obj) is not str:
        obj = str(obj).encode('utf8')
    return hashlib.md5(obj).hexdigest()[:16]


def get_model(model_name, reducer=None, parallel=False, **kwargs):
    try:
        model = models.__dict__[model_name](**kwargs)
    except KeyError:
        raise ValueError(f'Can\'t provide Model called "{model_name}"')

    if reducer is not None:
        reducer = get_model(reducer)
        model = TupleSequential(model, reducer)

    return model


def get_optimizer(name):
    try:
        return opt.__dict__[name]
    except KeyError:
        try:
            print('apex optimizer chosen')
            return apex.optimizers.__dict__[name]
        except KeyError:
            raise ValueError(f'Can\'t provide Optimizer called "{name}"')
        except NameError:
            raise ValueError(f'Can\'t provide Optimizer called "{name}"')


def get_loss_function(name):
    try:
        return loss_functions.__dict__[name]
    except KeyError:
        raise ValueError(f'Can\'t provide Loss Function called "{name}"')
