import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from downstream_task import DownstreamTask
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tiffile import imread
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from imageio import imwrite
import wandb


class ISPRS_Potsdam(DownstreamTask):
    def name(self):
        return 'ISPRS Potsdam Test 1'

    def n_classes(self):
        return 6

    def get_data(self):
        trn_data = potsdam_dataset(Path(__file__).parent / 'data/ISPRS_Potsdam', 'train')
        val_data = potsdam_dataset(Path(__file__).parent / 'data/ISPRS_Potsdam', 'val', patch_size=2000, stride=2000)

        trn_loader = DataLoader(trn_data, batch_size=8, pin_memory=True, num_workers=1, shuffle=True)
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

        for img, lbl, _ in train_loader:
            img = img.to(dev)
            features, low_level = backbone(img)
            feature_channels   = features.shape[1]
            low_level_channels = low_level.shape[1]
            break

        aspp = ASPP(feature_channels).to(dev)
        decoder = Decoder(low_level_channels, self.n_classes()).to(dev)

        model = DeepLab(backbone, aspp, decoder)
        for param in backbone.parameters():
            param.requires_grad = False
        opt = torch.optim.Adam(chain(aspp.parameters(), decoder.parameters()))

        for epoch in range(5):
            # self.train(model, train_loader, opt, epoch, dev)
            if epoch < 4: continue
            self.validate(model, val_loader, epoch, dev)

    def train(self, model, trn_loader, opt, epoch, dev):
        model.train()

        for img, lbl, _ in tqdm(trn_loader):
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
        img_dir = Path(__file__).parent / 'isprs_imgs'

        for i, (img, lbl, lbl_e) in enumerate(tqdm(val_loader)):
            img = img.to(dev)
            lbl_e = lbl_e.to(dev)

            pred = model(img)
            pred = torch.argmax(pred, dim=1)
            mask = (lbl_e != -1)
            correct = (pred == lbl_e)

            acc = correct[mask].float().mean().cpu()
            res.append(acc)

            if epoch == 4:
                mean   = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                std    = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                v_img  = (img[0].cpu() * std) + mean
                i_lbl  = lbl[0]
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

                imwrite(img_dir / f'{i}-_RGB.jpg', v_img.permute(1, 2, 0).to(torch.uint8))
                imwrite(img_dir / f'{i}-_GT.jpg',  (255.0 * v_lbl).permute(1, 2, 0).to(torch.uint8))
                imwrite(img_dir / f'{i}-{wandb.run.name}.jpg',  (255.0 * v_pred).permute(1, 2, 0).to(torch.uint8))

                oa = torch.stack(res).mean()
        print(f'Valid Epoch {epoch} -- OA: {oa:.4f}')
        wandb.log({f'{self.name()}/Accuracy': oa, f'_{self.name()}_epoch': epoch})


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
TRAIN_SCENES = [
    '2_10', '2_11', '2_12', '3_10', '3_11', '4_10', '4_11', '4_12', '5_10', '5_11', '5_12',
    '6_7', '6_8', '6_9', '6_10', '6_11', '6_12', '7_7', '7_8', '7_9', '7_10', '7_11', '7_12'
]


def potsdam_dataset(root, split, patch_size=512, stride=347):
    root = Path(root)
    scenes = []
    for scene in tqdm(list(sorted(root.glob('top_*_RGB.tif'))), desc='Loading Data...'):
        sceneid = '_'.join(scene.stem.split('_')[2:4])
        gt_path = scene.parent / f"{scene.stem[:-4]}_label.tif"
        gt_e_path = scene.parent / f"{scene.stem[:-4]}_label_noBoundary.tif"

        if (split == 'train') == (sceneid in TRAIN_SCENES):
                scenes.append(SingleTiffLoader(scene, gt_path, gt_e_path, patch_size, stride))

    return ConcatDataset(scenes)


GT_COLORS = {
    (  0,   0,   0): -1, # Ignored areas
    (255, 255, 255):  0, # Impervious surfaces
    (  0,   0, 255):  1, # Buildings
    (  0, 255, 255):  2, # Low vegetation
    (  0, 255,   0):  3, # Tree
    (255, 255,   0):  4, # Car
    (255,   0,   0):  5, # Clutter
}
class_labels = dict(enumerate(['Impervious Surfaces', 'Buildings', 'Low Vegetation', 'Tree', 'Car', 'Clutter']))


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


class SingleTiffLoader(Dataset):
    def __init__(self, scene_path, gt_path, gt_eroded_path, patch_size, stride, augment=False):
        self.augment = augment
        scene_cache = scene_path.with_suffix('.npy')
        if scene_cache.exists():
            self.top = np.load(scene_cache)
        else:
            self.top = imread(scene_path)
            np.save(scene_cache, self.top)

        self.gt        = cache_or_load_gt(gt_path)
        self.gt_eroded = cache_or_load_gt(gt_eroded_path)


        # Initialize views into the arrays
        self.views = []
        for y in np.arange(0, self.top.shape[0]-patch_size, stride):
            for x in np.arange(0, self.top.shape[1]-patch_size, stride):
                # This actually discards a small strip of data... fix it someday!
                self.views.append((
                    self.top[y:y+patch_size, x:x+patch_size],
                    self.gt[y:y+patch_size, x:x+patch_size],
                    self.gt_eroded[y:y+patch_size, x:x+patch_size]
                ))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __getitem__(self, idx):
        if self.augment:
            pass
        idx = idx % len(self.views)
        top, gt, gt_e = self.views[idx]
        gt = torch.from_numpy(gt)
        gt_e = torch.from_numpy(gt_e)
        top = torch.from_numpy(top.transpose(2, 0, 1).astype(np.float32))
        top = self.normalize(top)
        return top, gt, gt_e

    def __len__(self):
        return len(self.views)


if __name__ == '__main__':
    ISPRS_Potsdam().run_as_main()
