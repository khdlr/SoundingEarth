import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pickle
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import neighbors


import sys
root = Path(__file__).parent.parent
sys.path.append(str(root.absolute()))

from lib import get_model, get_loss_function, FullModelWrapper
from config import cfg

def evaluate_aid_few_shot(model, dev=None):
    if dev is None:
        dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    data_root = Path(__file__).parent / 'data/AID'

    images = list(data_root.glob('*/*.jpg'))
    images.sort()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    imgtransform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(192),
        transforms.ToTensor(), normalize
    ])
    torch.set_grad_enabled(False)

    img_encoder = model.img_encoder
    loss_function = model.loss_function

    data = []
    for img_path in tqdm(images):
        key = img_path.stem
        label = img_path.parent.name
        img = imgtransform(Image.open(img_path)).unsqueeze(0)
        img = img.to(dev)
        z_img = loss_function.distance_transform(img_encoder(img))[0].cpu().numpy()

        data.append(dict(
            key=key,
            label=label,
            z_img=z_img,
        ))

    key   = [d['key'] for d in data]
    label = [d['label'] for d in data]
    z_img = np.stack([d['z_img'] for d in data], axis=0)

    # Numeric Encoding for labels
    idx2lbl = np.unique(label)
    lbl2idx = {l: i for i, l in enumerate(idx2lbl)}
    label = np.array([lbl2idx[l] for l in label])
    class_weights = np.bincount(label) / len(label)

    trn_test = train_test_split(label, z_img, stratify=label, train_size=5*len(idx2lbl), random_state=42)
    lbl_trn, lbl_test, Z_trn, Z_test = trn_test

    seed = 42
    knn = neighbors.NearestCentroid()
    knn.fit(Z_trn, lbl_trn)
    prediction = knn.predict(Z_test)

    res = dict()
    precision, recall, fscore, _ = precision_recall_fscore_support(lbl_test, prediction, average='weighted')
    res['precision'] = precision
    res['recall'] = recall
    res['fscore'] = fscore
    res['accuracy'] = accuracy_score(lbl_test, prediction)

    print(res)
    wandb.log({
        f'AID FewShot/{metric.title()}': val for metric, val in res.items()
    })


if __name__ == '__main__':
    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    parser = ArgumentParser(description='Test Advance Data')
    parser.add_argument("model", type=Path, help='folder containing the trained model')
    args = parser.parse_args()

    cfg.merge_from_file(args.model / 'config.yml')
    cfg.freeze()

    run_id = cfg.RunId
    assert run_id != ''
    wandb.init(project='Audiovisual', resume=True, id=run_id)

    img_encoder   = get_model(cfg.ImageEncoder, reducer=cfg.ImageReducer,
        input_dim=3, output_dim=cfg.LatentDim, final_pool=False
    )
    snd_encoder   = get_model(cfg.SoundEncoder, reducer=cfg.SoundReducer,
        input_dim=1, output_dim=cfg.LatentDim, final_pool=True
    )
    loss_function = get_loss_function(cfg.LossFunction)(*cfg.LossArg)
    model = FullModelWrapper(img_encoder, snd_encoder, loss_function)
    model = model.to(dev)
    model.eval()

    model.load_state_dict(torch.load(args.model / 'checkpoints/100.pt'))
    evaluate_aid_few_shot(model)
