import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from data_loading import get_loader
import wandb
from pathlib import Path
from config import cfg


@torch.no_grad()
def calculate_embeddings(model, loader_type, device):
    loader = get_loader(mode=loader_type, batch_size=16, num_workers=4, max_samples=cfg.MaxSamples)
    keys, Z_snd, Z_img = [], [], []
    tf = model.loss_function.distance_transform
    for key, img, snd, snd_split, distance in tqdm(loader, 'Embeddings'):
        snd = snd.to(device)
        img = img.to(device)

        Z_snd.append(tf(model.snd_encoder(snd, snd_split)))
        Z_img.append(tf(model.img_encoder(img)))
        keys.append(key)

    keys = np.concatenate(keys)
    Z_img = torch.cat(Z_img)
    Z_snd = torch.cat(Z_snd)
    print(Z_snd.shape)

    return keys, Z_img, Z_snd


def evaluate(model, log_dir, dev=None):
    print('Doing final Evaluation...')
    if dev is None:
        dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    meta = pd.read_csv(Path(cfg.DataRoot) / 'metadata.csv')
    key2idx = {v: i for i, v in enumerate(meta.key)}

    def haversine_dist(key1, key2):
        # Haversine distance calculation
        idx1 = list(map(key2idx.get, key1))
        idx2 = list(map(key2idx.get, key2))
        lon1 = np.radians(meta.longitude.values[idx1]).reshape(-1, 1)
        lat1 = np.radians(meta.latitude.values[idx1]).reshape(-1, 1)

        lon2 = np.radians(meta.longitude.values[idx2]).reshape(1, -1)
        lat2 = np.radians(meta.latitude.values[idx2]).reshape(1, -1)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.square(np.sin(dlat/2)) + np.cos(lat1) * np.cos(lat2) * np.square(np.sin(dlon/2))
        c = 2 * np.arcsin(np.sqrt(a))
        dist = c * 6371  # distance in km

        return dist

    K, Z_img, Z_snd = calculate_embeddings(model, 'test', dev)

    # Match each sound against all audio
    distance_matrix = np.zeros([len(K), len(K)])
    for i in range(len(K)):
        dists = torch.linalg.norm(Z_img - Z_snd[i].unsqueeze(0), ord=2, dim=1).cpu().numpy()
        distance_matrix[[i]] = dists

    distance_matrix = distance_matrix / distance_matrix.mean(axis=0, keepdims=True)
    # Evaluate Img2Sound
    results = []
    for i_img, k_img in enumerate(tqdm(K, desc='Img2Sound')):
        df = pd.DataFrame(dict(
            k_snd = K,
            dist = distance_matrix[:, i_img]
        )).set_index('k_snd')

        df['rank'] = df.dist.rank()
        res = dict(
            rank=df.loc[k_img, 'rank'],
        )
        for topk in [10, 100, 1000]:
            matches = df.nsmallest(topk, 'dist').index
            res[f'km@{topk}'] = np.min(haversine_dist([k_img], matches))
        results.append(res)

    f = open(log_dir / 'results.txt', 'w')
    def tee(*msg):
        print(*msg)
        print(*msg, file=f)

    tee("=== Image2Sound Results ===")
    df = pd.DataFrame(results)
    df.to_csv(log_dir / 'img2sound.csv', index=False)

    i2s_metrics = {
        'R@100': (df['rank'] < 100).mean(),
        'R@500': (df['rank'] < 500).mean(),
        'R@1000': (df['rank'] < 1000).mean(),
        'D@100': df['km@100'].mean(),
        'D@1000': df['km@1000'].mean(),
        'Median Rank': df['rank'].median(),
    }

    for k, v in i2s_metrics.items():
        tee(k, v)

    x  = np.linspace(0, len(df), 3000)
    data = [[t, (df['rank'] < t).mean()] for t in x]
    table = wandb.Table(data=data, columns = ["Threshold", "Recall"])
    i2s_metrics["Recall Curve"] = wandb.plot.line(table, "Threshold", "Recall", title="I2S Recall @ Threshold")

    wandb.log({f'test/I2S {k}': i2s_metrics[k] for k in i2s_metrics})

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    tee(df.describe())

    # Evaluate Sound2Img
    results = []
    for i_snd, k_snd in enumerate(tqdm(K, desc='Sound2Img')):
        df = pd.DataFrame(dict(
            k_img = K,
            dist = distance_matrix[i_snd, :]
        )).set_index('k_img')

        df['rank'] = df.dist.rank()
        res = dict(
            rank=df.loc[k_snd, 'rank'],
        )
        for topk in [10, 100, 1000]:
            matches = df.nsmallest(topk, 'dist').index
            res[f'km@{topk}'] = np.min(haversine_dist([k_snd], matches))
        results.append(res)

    tee("=== Sound2Img Results ===")
    df = pd.DataFrame(results)
    df.to_csv(log_dir / 'sound2img.csv', index=False)

    s2i_metrics = {
        'R@100': (df['rank'] < 100).mean(),
        'R@500': (df['rank'] < 500).mean(),
        'R@1000': (df['rank'] < 1000).mean(),
        'D@100': df['km@100'].mean(),
        'D@1000': df['km@1000'].mean(),
        'Median Rank': df['rank'].median(),
    }

    for k, v in s2i_metrics.items():
        tee(k, v)

    x  = np.linspace(0, len(df), 3000)
    data = [[t, (df['rank'] < t).mean()] for t in x]
    table = wandb.Table(data=data, columns = ["Threshold", "Recall"])
    s2i_metrics["Recall Curve"] = wandb.plot.line(table, "Threshold", "Recall", title="S2I Recall @ Threshold")

    wandb.log({f'test/S2I {k}': s2i_metrics[k] for k in s2i_metrics})

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    tee(df.describe())


    f.close()
