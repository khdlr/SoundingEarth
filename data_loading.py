import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from config import cfg


class AporeeDataset(Dataset):
    def __init__(self, root, filter_fn=None, augment=False, max_samples=None):
        super().__init__()
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / 'metadata.csv')
        self.snd_meta = pd.read_csv(self.root / 'keyoffsets.csv')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        self.unnormalize = Unnormalize(mean=mean, std=std)
        self.maxlen = max_samples

        if augment and 'image' in cfg.AugmentationMode:
            self.imgtransform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(), normalize,
                # transforms.RandomErasing()
            ])
        else:
            self.imgtransform = transforms.Compose([
                transforms.ToTensor(), normalize
            ])

        if augment and 'sound' in cfg.AugmentationMode:
            self.sndtransform = transforms.Compose([
                transforms.RandomCrop(size=128, padding=3),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1),
                # transforms.RandomErasing()
            ])
        else:
            self.sndtransform = lambda x: x


        # join and merge
        img_present = set(int(f.stem) for f in (self.root).glob('images_small/*.jpg'))
        self.meta = self.meta[self.meta.key.isin(img_present)]
        self.meta = self.meta.merge(self.snd_meta, left_on='key', right_on='key', how='inner')
        self.meta = self.meta[self.meta.len != 0]
        if filter_fn:
            self.meta = self.meta[self.meta.key.apply(filter_fn)]
        self.meta = self.meta.reset_index(drop=True)
        self.h5 = None
        self.key2idx = {v: i for i, v in enumerate(self.meta.key)}
        self.augment = augment
        print('Number of Samples:', len(self.meta))

    def assert_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.root / 'spectrograms.h5', 'r')

    def get_asymmetric_sampler(self, batch_size, asymmetry):
        lon = np.radians(self.meta.longitude.values)
        lat = np.radians(self.meta.latitude.values)

        coords = np.stack([
            np.cos(lon) * np.cos(lat),
            np.sin(lon) * np.cos(lat),
            np.sin(lat),
        ], axis=1)

        return AsymmetricSampler(coords, asymmetry, batch_size)

    def get_batch(self, keys):
        true_indices = map(self.key2idx.get, keys)
        return self.collate([self[i] for i in true_indices])

    def collate(self, batch):
        key, img, audio, audio_split, v = zip(*batch)

        key = torch.tensor(key)
        img = torch.stack(img, dim=0)
        audio = torch.cat(audio, dim=0).unsqueeze(1)
        v = torch.stack(v, dim=0)
        audio_split = audio_split

        return key, img, audio, audio_split, v

    def __getitem__(self, idx):
        sample = self.meta.iloc[idx]
        key = sample.key

        img = Image.open(self.root / 'images_small' / f'{key}.jpg')
        img = self.imgtransform(img)

        self.assert_open()
        h5idx = sample.start
        audio_split = min(sample.len, self.maxlen)
        idx = h5idx + int(torch.randint(0, sample.len-audio_split+1, []))
        audio = self.h5['spectrogram'][idx:idx+audio_split]
        audio = torch.from_numpy(audio)
        audio = torch.cat([self.sndtransform(a.unsqueeze(0)) for a in audio])

        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        v = torch.from_numpy(np.stack([x, y, z])).float()

        return [key, img, audio, audio_split, v]

    def __len__(self):
        return len(self.meta)


class AsymmetricSampler(torch.utils.data.Sampler):
    def __init__(self, coords, asymmetry, batch_size):
        self.coords = coords
        self.asymmetry = asymmetry
        self.batch_size = batch_size
        self.knn = NearestNeighbors(n_neighbors=batch_size)
        self.knn.fit(self.coords)

    def sample_around(self, start):
        batch_idx = set([start])
        offset = self.asymmetry * self.coords[start]
        while len(batch_idx) < self.batch_size:
            X = torch.randn([1, 3]).numpy() + offset
            X = X / np.linalg.norm(X, ord='fro')
            _, candidates = self.knn.kneighbors(X)
            indices = (int(c) for c in candidates[0])
            batch_idx.add(next(i for i in indices if i not in batch_idx))

        return list(batch_idx)

    def rand(self):
        return int(torch.randint(0, self.coords.shape[0], []))

    def __iter__(self, ):
        for i in range(len(self)):
            if i % 2 == 0:
                start = self.rand()
                yield self.sample_around(start)
            else:
                yield [self.rand() for _ in range(self.batch_size)]

    def __len__(self, ):
        return self.coords.shape[0] // self.batch_size


def get_loader(batch_size, mode, num_workers=4, asymmetry=0, max_samples=100):
    FACTOR = 10
    filter_fn = {
        'train': lambda x: (x%FACTOR) not in (7, 5, 2),
        'val':   lambda x: (x%FACTOR) == 7,
        'test':  lambda x: (x%FACTOR) in (2, 5),
        'toy':  lambda x: (x%1000) in (2, 5),
        'all':   lambda x: True
    }.get(mode)
    is_train = (mode == 'train')
    dataset = AporeeDataset(cfg.DataRoot, filter_fn, augment=is_train, max_samples=max_samples)
    loader_args = dict(
        batch_size = batch_size,
        pin_memory = False,
        num_workers = num_workers,
        shuffle = is_train,
        collate_fn = dataset.collate,
        drop_last = True
    )
    if asymmetry != 0:
        loader_args['batch_sampler'] = dataset.get_asymmetric_sampler(batch_size, asymmetry)
        del loader_args['batch_size']
        del loader_args['shuffle']
        del loader_args['drop_last']
    return DataLoader(dataset, **loader_args)


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, -1, 1, 1)
        self.std = torch.tensor(std).reshape(1, -1, 1, 1)

    def __call__(self, tensor):
        return tensor.mul(self.std).add(self.mean)
