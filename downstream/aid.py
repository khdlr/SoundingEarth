import torch
import torch.nn.functional as F
from downstream_task import DownstreamTask
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

class AID(DownstreamTask):
    def name(self):
        return 'AID Finetuning'

    def n_classes(self):
        return 30

    def get_data(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        imgtransform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
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

if __name__ == '__main__':
    AID().run_as_main()
