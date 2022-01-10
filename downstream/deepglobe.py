import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from downstream_task import DownstreamTask
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from imageio import imread, imwrite
import wandb

from utils import RunningConfusionMatrix


class DeepGlobe(DownstreamTask):
    def name(self):
        return 'DeepGlobe'

    def n_classes(self):
        return 6

    def get_data(self):
        trn_data = deepglobe_dataset(Path(__file__).parent / 'data/DeepGlobe/train', 'train')
        val_data = deepglobe_dataset(Path(__file__).parent / 'data/DeepGlobe/train', 'val', patch_size=2448, stride=2448)
        print('Train Data:', len(trn_data))
        print('Val Data:', len(val_data))

        trn_loader = DataLoader(trn_data, batch_size=16, pin_memory=True, num_workers=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, pin_memory=True, num_workers=1)

        return trn_loader, val_loader

    def backbone_from_encoder(self, encoder):
        if type(encoder) is nn.Sequential and len(encoder) == 2:
            # Prepare true resnet for Sequential unpacking
            net = encoder[0][0]
            encoder = nn.Sequential(
                net.conv1, net.bn1, nn.ReLU(), net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4,
                nn.AdaptiveAvgPool2d([1, 1])
            )

        if type(encoder) is nn.Sequential and type(encoder[3]) is not nn.MaxPool2d:
            # tile2vec
            print("Tile2Vec Unpacking")
            return Backbone(
                nn.Sequential(*encoder[:6]),
                nn.Sequential(*encoder[6:8])
            )
        elif type(encoder) is nn.Sequential and len(encoder) == 9:
            # Pretrained/Randomly initialized ResNet, strip avg pooling
            if len(encoder[7]) == 2:
                # ResNet 18
                encoder[7][0].conv1.stride = (1, 1)
                encoder[7][0].conv1.dilation = (2, 2)
                encoder[7][0].conv1.padding  = (2, 2)
                encoder[7][0].downsample[0].stride = (1, 1)
                encoder[7][0].downsample[0].dilation = (2, 2)
            else:
                # ResNet 50
                encoder[7][0].conv2.stride = (1, 1)
                encoder[7][0].conv2.dilation = (2, 2)
                encoder[7][0].conv2.padding  = (2, 2)
                encoder[7][0].downsample[0].stride = (1, 1)
                encoder[7][0].downsample[0].dilation = (2, 2)

            return Backbone(
                nn.Sequential(*encoder[:5]),
                nn.Sequential(*encoder[5:8])
            )
        else:
            raise ValueError('Unpacking this encoder is not supported')

    def evaluate_model(self, encoder, dev=None):
        train_loader, val_loader = self.get_data()
        backbone = self.backbone_from_encoder(encoder)

        for img, lbl in train_loader:
            img = img.to(dev)
            features, low_level = backbone(img)
            feature_channels   = features.shape[1]
            low_level_channels = low_level.shape[1]
            break

        aspp = ASPP(feature_channels).to(dev)
        decoder = Decoder(low_level_channels, self.n_classes()).to(dev)

        model = DeepLab(backbone, aspp, decoder)
        # for param in backbone.parameters():
        #     param.requires_grad = False
        # opt = torch.optim.Adam(chain(aspp.parameters(), decoder.parameters()))
        opt = torch.optim.Adam(model.parameters())

        for epoch in range(5):
            self.train(model, train_loader, opt, epoch, dev)
            self.validate(model, val_loader, epoch, dev)

    def train(self, model, trn_loader, opt, epoch, dev):
        model.train()

        for img, lbl in tqdm(trn_loader):
            img = img.to(dev)
            lbl = lbl.to(dev, torch.long)

            prediction = model(img)
            loss = F.cross_entropy(prediction, lbl, ignore_index=-1)

            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def validate(self, model, val_loader, epoch, dev):
        model.eval()
        res = []
        viz = []
        img_dir = Path(__file__).parent / 'deepglobe_predictions'
        img_dir.mkdir(exist_ok=True)
        confusion_matrix = torch.zeros(self.n_classes() * self.n_classes(), dtype=torch.long, device=dev)

        for i, (img, lbl) in enumerate(tqdm(val_loader)):
            img = img.to(dev)
            lbl = lbl.to(dev)

            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            mask = (lbl != -1)
            correct = (pred == lbl)
            acc = correct[mask].float().mean().cpu()
            res.append(acc)

            idx = lbl[mask].flatten() + self.n_classes() * pred[mask].flatten()
            count = torch.bincount(idx, minlength = self.n_classes()*self.n_classes())
            confusion_matrix += count

            if epoch == 4:
                mean   = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                std    = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                v_img  = (img[0].cpu() * std) + mean
                i_lbl  = lbl[0].cpu()
                v_lbl  = torch.zeros_like(v_img)
                i_pred = pred[0]
                v_pred = torch.zeros_like(v_img)
                for color, key in GT_COLORS.items():
                    if key == -1: continue  # This shouldn't happen
                    v_pred[:, i_pred == key] = torch.tensor(color).reshape(-1, 1) / 255.0
                    v_lbl[:, i_lbl == key] = torch.tensor(color).reshape(-1, 1) / 255.0

                viz.append(torch.cat([
                    v_img, v_pred, v_lbl
                ], dim=1))

                rgbpath = img_dir / f'{i}-_RGB.jpg' 
                if not rgbpath.exists():
                    imwrite(rgbpath, (255.0 * v_img).permute(1, 2, 0).to(torch.uint8))

                gtpath = img_dir / f'{i}-_GT.jpg'
                if not gtpath.exists():
                    imwrite(gtpath,  (255.0 * v_lbl).permute(1, 2, 0).to(torch.uint8))

                imwrite(img_dir / f'{i}-{wandb.run.name}.jpg',  (255.0 * v_pred).permute(1, 2, 0).to(torch.uint8))

        oa = torch.stack(res).mean()
        CM = confusion_matrix.reshape(self.n_classes(), self.n_classes())
        i = torch.diag(CM)
        u = CM.sum(dim=1) + CM.sum(dim=0) - i
        miou = torch.mean(i.float() / u.float())

        print(f'Valid Epoch {epoch} -- OA: {oa:.4f}, mIoU: {miou:.4f}')
        # wandb.log({f'{self.name()}/Accuracy': oa, f'{self.name()}/mIoU': miou, f'_{self.name()}_epoch': epoch})


class DeepLab(nn.Module):
    def __init__(self, backbone, aspp, decoder):
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.decoder = decoder

    def forward(self, img):
        features, low_level = self.backbone(img)
        features = self.aspp(features)
        prediction = self.decoder(features, low_level)
        return F.interpolate(prediction, img.shape[-2], mode='bilinear', align_corners=False)


class Backbone(nn.Module):
    def __init__(self, initial_modules, final_modules):
        super().__init__()
        self.initial_modules = initial_modules
        self.final_modules = final_modules

    def forward(self, x):
        low_level = self.initial_modules(x)
        high_level = self.final_modules(low_level)
        return high_level, low_level

# === DeepLab Modules, taken from https://github.com/jfzhang95/pytorch-deeplab-xception ===
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride=16, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, num_classes, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
# === Below this: Data Loading ===
VALIDATION_IDS = set(int(x) for x in open(Path(__file__).parent / 'data/DeepGlobe/valid_ids.txt').readlines())


def deepglobe_dataset(root, split, patch_size=512, stride=347):
    root = Path(root)
    scenes = []
    for scene in tqdm(list(sorted(root.glob('*_sat.jpg'))), desc='Loading Data...'):
        sceneid = int(scene.stem.split('_')[0])
        gt_path = scene.parent / f'{sceneid}_mask.png'

        if (split == 'train') == (sceneid not in VALIDATION_IDS):
                scenes.append(SingleTileDataset(scene, gt_path, patch_size, stride))

    return ConcatDataset(scenes)


GT_COLORS = {
    (  0,   0,   0): -1, # Ignored areas
    (  0, 255, 255):  0, # Urban Land
    (255, 255,   0):  1, # Agriculture Land
    (255,   0, 255):  2, # Rangeland
    (  0, 255,   0):  3, # Forest Land
    (  0,   0, 255):  4, # Water
    (255, 255, 255):  5, # Barren Land
}
class_labels = dict(enumerate(['Urban Land', 'Agriculture Land', 'Rangeland', 'Forest Land', 'Water', 'Barren Land']))


def cache_or_load_gt(gt_path):
    gt_cache = gt_path.with_suffix('.npy')
    if gt_cache.exists():
        res = np.load(gt_cache)
    else:
        gt = imread(gt_path)
        gt = 255 * (gt > 127.5).astype(np.uint8)
        R, G, B = gt[:, :, 0], gt[:, :, 1], gt[:, :, 2]
        res = -np.ones(gt.shape[:2], np.int8)
        overall_mask = np.zeros(gt.shape[:2], np.bool)
        for (r, g, b), i in GT_COLORS.items():
            mask = (R == r) & (G == g) & (B == b)
            overall_mask |= mask
            res[mask] = i 

        pct = overall_mask.mean()
        if pct < 1:
            print(f'Got only {100*pct:.2f}% of the img @ {gt_path}')
            print('Missing:')
            print(gt[~overall_mask])

        np.save(gt_cache, res)
    return res


class SingleTileDataset(Dataset):
    def __init__(self, scene_path, gt_path, patch_size, stride, augment=False):
        self.augment = augment
        scene_cache = scene_path.with_suffix('.npy')
        if scene_cache.exists():
            self.top = np.load(scene_cache)
        else:
            self.top = imread(scene_path)
            np.save(scene_cache, self.top)

        self.gt = cache_or_load_gt(gt_path)

        # Initialize views into the arrays
        self.views = []
        for y in np.arange(0, self.top.shape[0]-patch_size+1, stride):
            for x in np.arange(0, self.top.shape[1]-patch_size+1, stride):
                # This actually discards a small strip of data... fix it someday!
                self.views.append((
                    self.top[y:y+patch_size, x:x+patch_size],
                    self.gt[y:y+patch_size, x:x+patch_size],
                ))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __getitem__(self, idx):
        if self.augment:
            pass
        idx = idx % len(self.views)
        top, gt = self.views[idx]
        gt = torch.from_numpy(gt)
        top = torch.from_numpy(top.transpose(2, 0, 1).astype(np.float32))
        top = self.normalize(top / 255.0)
        return top, gt

    def __len__(self):
        return len(self.views)


if __name__ == '__main__':
    DeepGlobe().run_as_main()
