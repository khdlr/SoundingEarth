import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepChanVese(nn.Module):
    def __init__(self, mu=0.25, dt=0.5, lambda1=1., lambda2=1., max_iter=100):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu
        self.dt = dt
        self.max_iter = max_iter

        carg = dict(padding=1, padding_mode='replicate', bias=False)
        self.variation_conv_1 = nn.Conv2d(1, 4, 3, **carg)
        self.variation_conv_1.weight = torch.nn.Parameter(torch.Tensor([
          [[0.0, 0.0, 0.0], [0.0, -1., 1.0], [0.0, 0.0, 0.0]], # phixp
          [[0.0, 0.0, 0.0], [-1., 1.0, 0.0], [0.0, 0.0, 0.0]], # phixn
          [[0.0, 0.0, 0.0], [0.0, -1., 0.0], [0.0, 1.0, 0.0]], # phiyp
          [[0.0, -1., 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], # phiyn
        ]).unsqueeze(1), requires_grad=False)

        self.variation_conv_0 = nn.Conv2d(1, 4, 3, **carg)
        self.variation_conv_0.weight = torch.nn.Parameter(torch.Tensor([
          [[0.0, -.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], # phiy0
          [[0.0, -.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], # phiy0
          [[0.0, 0.0, 0.0], [-.5, 0.0, 0.5], [0.0, 0.0, 0.0]], # phix0
          [[0.0, 0.0, 0.0], [-.5, 0.0, 0.5], [0.0, 0.0, 0.0]], # phix0
        ]).unsqueeze(1), requires_grad=False)

        self.variation_conv_k = nn.Conv2d(1, 4, 3, **carg)
        self.variation_conv_k.weight = torch.nn.Parameter(torch.Tensor([
          [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], # phix0
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # phix0
          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], # phiy0
          [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # phiy0
        ]).unsqueeze(1), requires_grad=False)

    def forward(self, image, init=None):
        if init is None:
            phi = self._checkerboard(image, 5)
        else:
            phi = init
        # Do Deep Net here :)
        # placeholder grayscale conversion:
        image = torch.mean(image, dim=1, keepdims=True)

        i = 0

        for i in range(self.max_iter):   # Fixed number of iterations
            # Calculate new level set
            phi = self._calculate_variation(image, phi)

        return phi

    def _delta(self, x, eps=1.):
        return eps / (eps**2 + x**2)

    def _calculate_averages(self, image, Hphi):
        I = Hphi
        O = 1. - Hphi

        avginside  = spatial_sum(image * I) / torch.clip(spatial_sum(I), min=1)
        avgoutside = spatial_sum(image * O) / torch.clip(spatial_sum(O), min=1)

        return avginside, avgoutside

    def _checkerboard(self, image, square_size):
        """Generates a checkerboard level set function.
        According to Pascal Getreuer, such a level set function has fast convergence.
        """
        cb = chan_vese._cv_checkerboard(image.shape[2:], 5)
        
        yv = torch.arange(image.shape[2], device=image.device).reshape(image.shape[2], 1)
        xv = torch.arange(image.shape[3], device=image.device).reshape(1, image.shape[3])
        init = (torch.sin(pi/square_size*yv) *
                torch.sin(pi/square_size*xv))
        init = init.unsqueeze(0).unsqueeze(0)
        return init.expand([image.shape[0], 1, image.shape[2], image.shape[3]])

    def _calculate_variation(self, image, phi):
        eta = 1e-16
        phi_v1 = self.variation_conv_1(phi)
        phi_v0 = self.variation_conv_0(phi)
        
        C = torch.pow(eta + torch.square(phi_v1) + torch.square(phi_v0), -0.5)
        k_mul = self.variation_conv_k(phi)
        K = channel_sum(k_mul * C)

        Hphi = (phi > 0).float()
        c1, c2 = self._calculate_averages(image, Hphi)

        difference_from_average_term = (
            -self.lambda1 * torch.square(image - c1) +
             self.lambda2 * torch.square(image - c2)
        )
   
        delta_phi = self._delta(phi)

        new_phi = (phi +
          (self.dt * delta_phi *
          (self.mu * K + difference_from_average_term)
        ))

        return new_phi / (1 + self.mu * self.dt * delta_phi * channel_sum(C))
