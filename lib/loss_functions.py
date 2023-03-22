import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, img, snd, v=None):
        img = rearrange(img, '(i a) d -> i a d', a=1)
        snd = rearrange(snd, '(i a) d -> i a d', i=1)

        distance = torch.linalg.norm(img - snd, ord=2, dim=-1)
        d_true = distance.diagonal()

        d1 = torch.mean(F.relu(d_true.unsqueeze(0) - distance + 1.0))
        d2 = torch.mean(F.relu(d_true.unsqueeze(1) - distance + 1.0))

        return 0.5 * (d1 + d2)


class NaiveTriplet(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, img, snd, v=None):
        d_true = torch.linalg.norm(img - snd, ord=2, dim=-1)
        d_false = torch.linalg.norm(img - torch.roll(snd, 1, dims=0), ord=2, dim=-1)

        return torch.mean(F.relu(d_true - d_false + 1.0))


class BarlowTwins(nn.Module):
    def __init__(self, lambda_=5e-3, reg=0.1):
        super().__init__()
        self.lambda_ = lambda_
        self.reg_strength = reg

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, img, snd, v=None):
        N, D = img.shape

        concat = torch.cat([img, snd], dim=0)
        mu = concat.mean(0, keepdims=True)
        sigma = concat.std(0, keepdims=True)

        img_norm = (img - mu) / sigma
        snd_norm = (snd - mu) / sigma
        c = torch.mm(img_norm.t(), snd_norm) / N

        with torch.no_grad():
            tgt = torch.eye(D, device=c.device)
            factors = torch.maximum(
                    torch.eye(D, device=c.device),
                    self.lambda_ * torch.ones(D, D, device=c.device)
            )

        c_diff = (c - tgt).pow(2) * factors
        loss = c_diff.sum()

        implied_dist = torch.distributions.Normal(mu, sigma)
        target_dist = torch.distributions.Normal(mu.new_zeros([]), sigma.new_ones([]))
        reg = torch.mean(torch.distributions.kl_divergence(implied_dist, target_dist))

        return loss + self.reg_strength * reg


class SymmetricSimCLR(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau
        self.g = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, img, snd, v=None):
        img = self.g(img)
        snd = self.g(snd)

        img = rearrange(img, '(i a) d -> i a d', a=1)
        snd = rearrange(snd, '(i a) d -> i a d', i=1)

        sim = F.cosine_similarity(img, snd, dim=-1) / self.tau

        logsoftmax0 = torch.log_softmax(sim, dim=0)
        logsoftmax1 = torch.log_softmax(sim, dim=1)
        loss = -torch.mean(torch.diag(logsoftmax0)) - torch.mean(torch.diag(logsoftmax1))
        return loss


class SimCLRLoss(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau
        self.g = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, img, snd, v=None):
        img = self.g(img)
        snd = self.g(snd)

        img = rearrange(img, '(i a) d -> i a d', a=1)
        snd = rearrange(snd, '(i a) d -> i a d', i=1)

        sim = F.cosine_similarity(img, snd, dim=-1) / self.tau

        logsoftmax0 = torch.log_softmax(sim, dim=0)
        loss = -torch.mean(torch.diag(logsoftmax0))
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, x, y, margin=torch.ones([])):
        x = rearrange(x, '(i a) d -> i a d', a=1)
        y = rearrange(y, '(i a) d -> i a d', i=1)

        sim = F.cosine_similarity(x, y, dim=-1)
        logsoftmax = torch.log_softmax(sim / self.tau, dim=1)
        loss = -torch.mean(torch.diag(logsoftmax))
        return loss


class SymmetricCLWithDistillation(nn.Module):
    def __init__(self, tau=0.1, nu=0.1):
        super().__init__()
        self.tau = tau
        self.nu = nu

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, x, y, v=None):
        sim = F.cosine_similarity(x, y, dim=-1) / self.tau
        if v is not None:
            with torch.no_grad():
                dotprods = torch.einsum('ad,id->ai', v, v)
                target0 = F.softmax(dotprods / self.nu, dim=0)
                target1 = target0.t()
        logsoftmax0 = torch.log_softmax(sim, dim=0)
        logsoftmax1 = torch.log_softmax(sim, dim=1)

        ce0 = -torch.einsum('ai,ai->i', logsoftmax0, target0)
        ce1 = -torch.einsum('ai,ai->a', logsoftmax1, target1)

        loss = torch.mean(ce0) + torch.mean(ce1)
        return loss


class SymmetricCL(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, img, snd, v=None):
        img = rearrange(img, '(i a) d -> i a d', a=1)
        snd = rearrange(snd, '(i a) d -> i a d', i=1)

        sim = F.cosine_similarity(img, snd, dim=-1) / self.tau

        logsoftmax0 = torch.log_softmax(sim, dim=0)
        logsoftmax1 = torch.log_softmax(sim, dim=1)

        loss = -torch.mean(torch.diag(logsoftmax0)) - torch.mean(torch.diag(logsoftmax1))
        return loss


class SymmetricCLWithMemory(nn.Module):
    def __init__(self, tau=0.1, batches_to_cache=128):
        super().__init__()
        self.tau = tau
        self.img_cache = []
        self.snd_cache = []
        self.batches_to_cache = batches_to_cache

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, img, snd, v=None):
        if self.training:
            img_ext = torch.cat([img, *self.img_cache], dim=0)
            snd_ext = torch.cat([snd, *self.snd_cache], dim=0)

            img_cur = rearrange(img, '(i a) d -> i a d', a=1)
            snd_cur = rearrange(snd, '(i a) d -> i a d', i=1)

            img_ext = rearrange(img_ext, '(i a) d -> i a d', a=1)
            snd_ext = rearrange(snd_ext, '(i a) d -> i a d', i=1)

            sim_img  = F.cosine_similarity(img_cur, snd_ext, dim=-1) / self.tau
            loss_img = -torch.mean(torch.diag(torch.log_softmax(sim_img, dim=1)))

            sim_snd = F.cosine_similarity(snd_cur, img_ext, dim=-1) / self.tau
            loss_snd = -torch.mean(torch.diag(torch.log_softmax(sim_snd, dim=0)))

            loss = loss_img + loss_snd

            # Handle caching
            self.img_cache.append(img.detach())
            self.snd_cache.append(snd.detach())
            if len(self.img_cache) > self.batches_to_cache:
                self.img_cache = self.img_cache[1:]
                self.snd_cache = self.snd_cache[1:]
        else:
            img_cur = rearrange(img, '(i a) d -> i a d', a=1)
            snd_cur = rearrange(snd, '(i a) d -> i a d', i=1)

            sim = F.cosine_similarity(img_cur, snd_cur, dim=-1) / self.tau
            logsoftmax0 = torch.log_softmax(sim, dim=0)
            logsoftmax1 = torch.log_softmax(sim, dim=1)
            loss = -torch.mean(torch.diag(logsoftmax0)) - torch.mean(torch.diag(logsoftmax1))


        return loss


class SymmetricCLWithMemoryAndDistillation(nn.Module):
    def __init__(self, tau=0.1, nu=0.00000001, batches_to_cache=128):
        super().__init__()
        self.tau = tau
        self.nu = nu
        self.img_cache = []
        self.snd_cache = []
        self.loc_cache = []
        self.batches_to_cache = batches_to_cache

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, img, snd, loc=None):
        img = self.distance_transform(img)
        snd = self.distance_transform(snd)
        if self.training:
            loc_ext = torch.cat([loc, *self.loc_cache], dim=0)
            img_ext = torch.cat([img, *self.img_cache], dim=0)
            snd_ext = torch.cat([snd, *self.snd_cache], dim=0)
        else:
            loc_ext = loc
            img_ext = img
            snd_ext = snd

        with torch.no_grad():
            dotprods = torch.einsum('bd,cd->bc', loc, loc_ext)
            target   = F.softmax(dotprods / self.nu, dim=1)

        sim_img = torch.log_softmax(torch.einsum('bd,cd->bc', img, snd_ext), dim=1)
        sim_snd = torch.log_softmax(torch.einsum('bd,cd->bc', snd, img_ext), dim=1)

        loss_img = -torch.einsum('bc,bc->b', sim_img, target)
        loss_snd = -torch.einsum('bc,bc->b', sim_snd, target)

        loss = torch.mean(loss_img) + torch.mean(loss_snd)

        if self.training:
            # Handle caching
            self.img_cache.append(img.detach())
            self.snd_cache.append(snd.detach())
            self.loc_cache.append(loc.detach())
            if len(self.img_cache) > self.batches_to_cache:
                self.img_cache = self.img_cache[1:]
                self.snd_cache = self.snd_cache[1:]
                self.loc_cache = self.loc_cache[1:]

        return loss

###### Loss Functions below are old code and won't work currently

class EuclideanContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, x, y, margin=torch.ones([])):
        sim = -torch.linalg.norm(x - y, ord=2, dim=-1)
        logsoftmax = torch.log_softmax(sim, dim=1)
        loss = -torch.mean(torch.diag(logsoftmax))
        return loss


class LearnedContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = nn.Parameter(torch.ones([]))

    def distance_transform(self, x, dim=1):
        return x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)

    def forward(self, x, y, margin=torch.ones([])):
        sim = F.cosine_similarity(x, y, dim=-1)
        logsoftmax = torch.log_softmax(sim / self.tau, dim=1)
        loss = -torch.mean(torch.diag(logsoftmax))
        return loss


class DeepHashingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distance_transform(self, x, dim=1):
        return torch.sign(x)

    def forward(self, x, y, margin=torch.ones([])):
        # Generate Hash-Like Codes
        R = torch.arange(x.shape[0])
        x = x[R, R]
        y = y[R, R]

        q_a = torch.tanh(x / 32)
        q_p = torch.tanh(y / 32)
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

    def distance_transform(self, x, dim=1):
        return x

    def forward(self, x, y, margin=None):
        # Positive sample
        distance = torch.linalg.norm(x - y, ord=2, dim=-1)
        distance_logit = distance - self.logit_offset
        target = 1. - torch.eye(distance_logit.shape[0], device=distance_logit.device)
        loss = F.binary_cross_entropy_with_logits(distance_logit, target)

        return loss
