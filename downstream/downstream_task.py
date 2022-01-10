import sys
from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import sys
root = Path(__file__).parent.parent
sys.path.append(str(root.absolute()))

from lib import get_model, get_loss_function, FullModelWrapper
from lib.models import tile2vec_resnet
from config import cfg


class DownstreamTask(metaclass=ABCMeta):
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def n_classes(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data(self):
        raise NotImplementedError()


    def run_as_main(self):
        dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        parser = ArgumentParser(description='Test Advance Data')
        parser.add_argument("model", type=str, help='folder containing the trained model')
        parser.add_argument("backbone", type=str, choices=['resnet18', 'resnet50'],
                help='backbone model to use', default='resnet18', nargs='?')
        args = parser.parse_args()

        if args.model == 'imagenet':
            print('Using imagenet weights')
            runid = f'imagenet-rn{args.backbone[-2:]}'
            wandb.init(project='Audiovisual', name=f'ImageNet RN{args.backbone[-2:]}',
                   resume='allow', id=runid)
            encoder = from_resnet(args.backbone, pretrained=True)
        elif args.model == 'random':
            print('Using random weights')
            runid = f'random-rn{args.backbone[-2:]}'
            encoder = from_resnet(args.backbone, pretrained=False)
            wandb.init(project='Audiovisual', name=f'Random RN{args.backbone[-2:]}',
                    resume='allow', id=runid)
        elif args.model == 'tile2vec':
            print('Using Tile2Vec weights')
            encoder = from_tile2vec()
            runid = f'tile2vec'
            wandb.init(project='Audiovisual', name=f'Tile2Vec RN18',
                    resume='allow', id=runid)
        else:
            print('Using pre-trained weights')
            cfg.merge_from_file(Path(args.model) / 'config.yml')
            cfg.freeze()

            img_encoder   = get_model(cfg.ImageEncoder, reducer=cfg.ImageReducer,
                input_dim=3, output_dim=cfg.LatentDim, final_pool=False
            )
            snd_encoder   = get_model(cfg.SoundEncoder, reducer=cfg.SoundReducer,
                input_dim=1, output_dim=cfg.LatentDim, final_pool=True
            )
            loss_function = get_loss_function(cfg.LossFunction)(*cfg.LossArg)
            full_model = FullModelWrapper(img_encoder, snd_encoder, loss_function)
            full_model = full_model.to(dev)
            full_model.load_state_dict(torch.load(Path(args.model) / 'checkpoints/latest.pt', map_location=dev))

            encoder = nn.Sequential(full_model.img_encoder, Normalizer(full_model.loss_function))
            assert cfg.RunId != ''
            wandb.init(project='Audiovisual', resume='must', id=cfg.RunId)
        encoder = encoder.to(dev)
        self.evaluate_model(encoder, dev)


    @torch.no_grad()
    def validate(self, model, val_loader, epoch, dev):
        print(f'Valid Epoch {epoch}')
        model.eval()
        res = []
        for img, label in tqdm(val_loader):
            img = img.to(dev)
            label = label.to(dev)

            prediction = model(img)
            correct = (torch.argmax(prediction, dim=1) == label)
            res.append(correct.cpu())

        res = torch.cat(res).float()
        oa = res.mean()
        print(f'Epoch {epoch}: {oa:.4f}')
        wandb.log({f'{self.name()}/Accuracy': oa, f'_{self.name()}_epoch': epoch})


    def train(self, model, trn_loader, opt, epoch, dev):
        print(f'Train Epoch {epoch}')
        model.train()
        for img, label in tqdm(trn_loader):
            img = img.to(dev)
            label = label.to(dev)

            prediction = model(img)
            loss = F.cross_entropy(prediction, label)
            opt.zero_grad()
            loss.backward()
            opt.step()


    def evaluate_model(self, encoder, dev=None):
        # make a copy so we don't change the original model params
        encoder = copy.deepcopy(encoder)
        if dev is None:
            dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

        # Add final layer
        encoder = encoder.to(dev)
        out_shp = encoder(torch.zeros(1, 3, 64, 64, device=dev)).shape

        last_layer = nn.Linear(out_shp[1], self.n_classes())
        model = nn.Sequential(
            encoder,
            nn.Flatten(),
            last_layer,
        )
        model = model.to(dev)
        trn_data, val_data = self.get_data()
        epoch = 0

        for param in model.parameters():
            param.requires_grad = True
        opt = torch.optim.Adam(model.parameters())
        for epoch in range(1, 6):
            self.train(model, trn_data, opt, epoch, dev)
            self.validate(model, val_data, epoch, dev)


def from_resnet(resnet_type, pretrained=False):
    resnet_class = getattr(tv_models, resnet_type)
    resnet = resnet_class(pretrained=pretrained)
    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        nn.AdaptiveAvgPool2d([1, 1]),
    )


class SnoopShape(nn.Module):
    def __init__(self, name=''):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.shape)
        return x


def from_tile2vec():
    net = tile2vec_resnet.ResNet18()
    net.load_state_dict(torch.load('logs/Tile2Vec/naip_trained.ckpt'))
    net.conv1.weight = torch.nn.Parameter(net.conv1.weight[:, :3])
    net.conv1.in_channels = 3
    net.supervised = False
    encoder = nn.Sequential(
        net.conv1, net.bn1, nn.ReLU(),
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
        net.layer5,
        nn.AdaptiveAvgPool2d([1, 1])
    )
       
    return encoder


class Normalizer(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.loss_fn.distance_transform(x)
