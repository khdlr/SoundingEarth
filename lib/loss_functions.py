import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=-1):
        return x

    def forward(self, x, y, margin=torch.ones([])):
        distance = torch.linalg.norm(x - y, ord=2, dim=-1)
        d_true = distance.diagonal()

        d1 = torch.mean(F.relu(d_true.unsqueeze(0) - distance + margin))
        d2 = torch.mean(F.relu(d_true.unsqueeze(1) - distance + margin))

        return 0.5 * (d1 + d2)


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=-1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, x, y, margin=torch.ones([])):
        sim = F.cosine_similarity(x, y, dim=2)
        logsoftmax = torch.log_softmax(sim, dim=1)
        loss = -torch.mean(torch.diag(logsoftmax))
        return loss


class DeepHashingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=-1):
        return torch.tanh(x)

    def forward(self, x, y, margin=torch.ones([])):
        # Generate Hash-Like Codes
        q_a = torch.tanh(z_img / 32)
        q_p = torch.tanh(z_snd / 32)
        q_n = torch.roll(q_p, 1, dims=0)

        # Generate True Hashes
        h_a = torch.sign(q_a)
        h_p = torch.sign(q_p)

        triplet_loss = F.triplet_margin_loss(q_a, q_p, q_n)
        triplet_regularized_loss = (
            F.l1_loss(q_a, h_a) +
            F.l1_loss(q_p, h_p)
        )

        return triplet_loss + 0.01 * triplet_regularized_loss


class SigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_offset = nn.Parameter(torch.ones([]))

    def distance_transform(self, x, dim=-1):
        return x

    def forward(self, z_img, z_snd, margin=None):
        # Positive sample
        dist_pos = torch.linalg.norm(z_img - z_snd, ord=2, dim=1).unsqueeze(1)
        logit_pos = dist_pos - self.logit_offset
        nll_pos = -torch.mean(F.logsigmoid(-logit_pos))

        # Negative sample
        z_snd_neg = torch.roll(z_snd, 1, dims=0)
        dist_neg = torch.linalg.norm(z_img - z_snd_neg, ord=2, dim=1).unsqueeze(1)
        logit_neg = dist_neg - self.logit_offset
        nll_neg = -torch.mean(F.logsigmoid(logit_neg))

        return nll_pos + nll_neg
