import sys
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pickle
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models

import sys
root = Path(__file__).parent.parent
sys.path.append(str(root.absolute()))

from lib import get_model, get_loss_function, FullModelWrapper
from config import cfg


def get_data():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    imgtransform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), 2).squeeze(0))
    ])

    all_data = ImageFolder('downstream/data/AID/', transform=imgtransform)

    L = len(all_data)
    trn_index = torch.arange(0, L, 2)
    val_index   = torch.arange(1, L, 2)
    trn_data = Subset(all_data, trn_index)
    val_data = Subset(all_data, val_index)

    trn_loader = DataLoader(trn_data, batch_size=8, pin_memory=True, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, pin_memory=True, num_workers=4)

    return trn_loader, val_loader


class Normalizer(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.loss_fn.distance_transform(x)


@torch.no_grad()
def validate(model, val_loader, epoch, dev):
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
    wandb.log({f'AID/Accuracy': oa, '_aid_epoch': epoch})


def train(model, trn_loader, opt, epoch, dev):
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


def evaluate_aid(model, dev):
    evaluate_model(copy.deepcopy(model.img_encoder), dev)


def evaluate_model(encoder, dev=None):
    if dev is None:
        dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    # Add final layer
    encoder = encoder.to(dev)
    out_shp = encoder(torch.zeros(1, 3, 64, 64, device=dev)).shape

    last_layer = nn.Linear(out_shp[1], 30)
    model = nn.Sequential(
        encoder,
        nn.Flatten(),
        last_layer,
    )
    model = model.to(dev)
    trn_data, val_data = get_data()
    epoch = 0

    for param in model.parameters():
        param.requires_grad = True
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(1, 6):
        train(model, trn_data, opt, epoch, dev)
        validate(model, val_data, epoch, dev)


if __name__ == '__main__':
    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    parser = ArgumentParser(description='Test Advance Data')
    parser.add_argument("model", type=str, help='folder containing the trained model')
    parser.add_argument("backbone", type=str, choices=['resnet18', 'resnet50'], help='backbone model to use', default='resnet18')
    args = parser.parse_args()

    if args.model == 'imagenet':
        print('Using imagenet weights')
        wandb.init(project='Audiovisual', name='ImageNet')
        encoder = from_resnet(args.backbone, pretrained=True)
    elif args.model == 'random':
        print('Using random weights')
        encoder = from_resnet(args.backbone, pretrained=False)
        wandb.init(project='Audiovisual', name='Random')
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
        full_model.load_state_dict(torch.load(Path(args.model) / 'checkpoints/latest.pt'))

        encoder = nn.Sequential(full_model.img_encoder, Normalizer(full_model.loss_function))
        assert cfg.RunId != ''
        wandb.init(project='Audiovisual', resume=True, id=cfg.RunId)

    evaluate_model(encoder)
