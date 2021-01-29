import torch
import numpy as np
import rasterio as rio
from pathlib import Path
from lib import md5

from config import cfg


def _isval(gt_path):
    year = int(gt_path.stem[:4])

    return year >= 2020


def get_loader(batch_size, data_threads, mode):
    data = UC1Dataset(mode=mode)
    if mode == 'train':
        data = Augment(data)

    return torch.utils.data.DataLoader(data,
        batch_size=batch_size,
        num_workers=data_threads,
        pin_memory=True
    )


class UC1Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.root = Path('/data/aicore/uc1/data/aicore')
        self.cachedir = self.root.parent / 'cache'
        self.cachedir.mkdir(exist_ok=True)
        self.confighash = md5(cfg.Bands)

        self.gts = sorted(list(self.root.glob('ground_truth/*/*/*_30m.tif')))
        if mode == 'train':
            self.gts = [g for g in self.gts if not _isval(g)]
        else:
            self.gts = [g for g in self.gts if _isval(g)]

    def __getitem__(self, idx):
        path = self.gts[idx]
        *_, site, date, gtname = path.parts
        gt_cache = self.cachedir / f'{site}_{date}.pt'
        ref_cache = self.cachedir / f'{site}_{date}_{self.confighash}.pt'

        if gt_cache.exists():
            gt = torch.load(gt_cache)
        else:
            with rio.open(path) as raster:
                gt = raster.read(1)
            gt = torch.from_numpy(gt).to(torch.bool)
            torch.save(gt, gt_cache)
        gt = gt.to(torch.float32)

        if ref_cache.exists():
            ref = torch.load(ref_cache)
        else:
            ref_root = self.root / 'reference_data' / site / date / '30m'

            ref = []
            for band in cfg.Bands:
                try:
                    with rio.open(ref_root / f'{band}.tif') as raster:
                        ref.append(raster.read(1).astype(np.uint8))
                except rio.errors.RasterioIOError:
                    print(f'RasterioIOError when opeining {ref_root}/{band}.tif')
                    return None
            ref = torch.from_numpy(np.stack(ref, axis=-1)).to(torch.uint8)
            torch.save(ref, ref_cache)
        ref = ref.to(torch.float32) / 255

        return ref, gt

    def __len__(self):
        return len(self.gts)


def _nested_apply(fun, arg):
    t = type(arg)
    if t in [tuple, list, set]:
        return t(_nested_apply(fun, x) for x in arg)
    else:
        return fun(arg)


def _augment_transform(opflags):
    flipx, flipy, transpose = opflags
    def augmentation(field):
        if type(field) is not torch.Tensor or field.ndim < 3:
            # Not spatial! -> no augmentation
            return field
        if transpose:
            field = field.transpose(-1, -2)
        if flipx or flipy:
            dims = []
            if flipy: dims.append(-2)
            if flipx: dims.append(-1)
            field = torch.flip(field, dims)
        return field
    return augmentation


class Augment(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for el in self.dataset:
            opflags = torch.randint(2, [3])
            yield _nested_apply(_augment_transform(opflags), el)


if __name__ == '__main__':
    from tqdm import tqdm

    print('Train..')
    ds = UC1Dataset(mode='train')
    print('First pass')
    for _ in tqdm(ds):
        pass
    print('Second pass')
    for _ in tqdm(ds):
        pass

    print('Val..')
    ds = UC1Dataset(mode='val')
    print('First pass')
    for _ in tqdm(ds):
        pass
    print('Second pass')
    for _ in tqdm(ds):
        pass
