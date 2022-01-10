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
from sklearn.metrics import precision_recall_fscore_support

import sys
root = Path(__file__).parent.parent
sys.path.append(str(root.absolute()))

from lib import get_model, get_loss_function, FullModelWrapper
from config import cfg

LOW  = np.exp(-15 / 10)
HIGH = np.exp(5 / 10)

def evaluate_advance(model, dev=None):
    if dev is None:
        dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    img_encoder = model.img_encoder
    snd_encoder = model.snd_encoder
    loss_function = model.loss_function

    data_root = Path(__file__).parent / 'data/advance'

    images = list((data_root / 'vision').glob('*/*.jpg'))
    images.sort()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    imgtransform = transforms.Compose([
        transforms.ToTensor(), normalize
    ])
    torch.set_grad_enabled(False)

    data = []
    for img_path in tqdm(images):
        key = img_path.stem
        label = img_path.parent.name

        snd_path = data_root / 'sound' / label / f'{key}.npy'
        img = imgtransform(Image.open(img_path))
        img = img.to(dev)
        img = F.avg_pool2d(img.unsqueeze(0), 2)

        snd = np.load(snd_path).astype(np.float32)
        snd = np.clip((np.exp(snd / 10) - LOW) / (HIGH - LOW), 0, 1)
        snd = torch.from_numpy(snd).to(dev)
        snd = snd.unsqueeze(0).unsqueeze(1)
        snd_splits = (1, )

        z_img = loss_function.distance_transform(img_encoder(img))[0].cpu().numpy()
        z_snd = loss_function.distance_transform(snd_encoder(snd, snd_splits))[0].cpu().numpy()

        data.append(dict(
            key=key,
            label=label,
            z_img=z_img,
            z_snd=z_snd
        ))


    key   = [d['key'] for d in data]
    label = [d['label'] for d in data]
    z_img = np.stack([d['z_img'] for d in data], axis=0)
    z_snd = np.stack([d['z_snd'] for d in data], axis=0)

    # Numeric Encoding for labels
    idx2lbl = np.unique(label)
    lbl2idx = {l: i for i, l in enumerate(idx2lbl)}
    label = np.array([lbl2idx[l] for l in label])
    class_weights = np.bincount(label) / len(label)

    results = []

    trnval_test = train_test_split(key, label, z_img, z_snd, stratify=label, test_size=0.2, random_state=42)
    key_trnval, key_test, lbl_trnval, lbl_test, Z_img_trnval, Z_img_test, Z_snd_trnval, Z_snd_test = trnval_test
    for data_mode in ['image', 'sound', 'mean', 'concat']:
    # for data_mode in ['concat']:
        print(f'Taking {data_mode} as data basis.')
        for i in tqdm(range(5)):
            seed = 42 * i

            trn_val = train_test_split(key_trnval, lbl_trnval, Z_img_trnval, Z_snd_trnval,
                            stratify=lbl_trnval, test_size=0.1/0.8, random_state=seed)
            key_trn, key_val, lbl_trn, lbl_val, Z_img_trn, Z_img_val, Z_snd_trn, Z_snd_val = trn_val

            Z_trn = dict(
                image=Z_img_trn,
                sound=Z_snd_trn,
                mean=(Z_img_trn + Z_snd_trn) / 2,
                concat=np.concatenate([Z_img_trn, Z_snd_trn], axis=1)
            )[data_mode]
            Z_val = dict(
                image=Z_img_val,
                sound=Z_snd_val,
                mean=(Z_img_val + Z_snd_val) / 2,
                concat=np.concatenate([Z_img_val, Z_snd_val], axis=1)
            )[data_mode]
            Z_test = dict(
                image=Z_img_test,
                sound=Z_snd_test,
                mean=(Z_img_test + Z_snd_test) / 2,
                concat=np.concatenate([Z_img_test, Z_snd_test], axis=1)
            )[data_mode]

            if 'Contrastive' in cfg.LossFunction:
                penalty = 'none'
            else:
                penalty = 'l2'
            lr = LogisticRegression(multi_class='multinomial', max_iter=500000, penalty=penalty, n_jobs=1)
            lr.fit(Z_trn, lbl_trn)
            prediction = lr.predict(Z_test)

            res = dict(data=data_mode)
            precision, recall, fscore, _ = precision_recall_fscore_support(lbl_test, prediction, average='weighted')
            res['precision'] = precision
            res['recall'] = recall
            res['fscore'] = fscore

            results.append(res)

    df = pd.DataFrame(results)
    pd.options.display.float_format = lambda x: f'{100*x:.2f}%'

    # metrics = dict(df[df.data == 'concat'].mean())
    # wandb.log({
    #     f'Advance/{metric.title()}': val for metric, val in metrics.items()
    # })

    for data_mode, data in df.groupby('data'):
        metrics = dict(data.mean())
        wandb.log({
            f'AdvanceDetailed/{data_mode}_{metric.title()}': val for metric, val in metrics.items()
        })

    res = df.melt(id_vars=['data']).groupby(['data', 'variable']).agg(['mean', 'std']).T
    print(res['concat'][['precision', 'recall', 'fscore']])



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

    model.load_state_dict(torch.load(args.model / 'checkpoints/best.pt', map_location=dev))
    evaluate_advance(model)
