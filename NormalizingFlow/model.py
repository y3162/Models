import torch 
import torch.nn as nn
import torch.nn.functional as F

class RealNVP(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.scale_net = nn.Sequential(
            nn.Linear(latent_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim // 2),
            nn.Tanh(),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(latent_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim // 2),
        )

    def _chunk_forward(z):
        D = z.shape[-1]
        d = D // 2
        return z[..., :d], z[..., d:]

    def forward(self, z):
        z1, z2 = RealNVP._chunk_forward(z)
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        z2_new = z2 * torch.exp(s) + t
        z_new = torch.cat([z1, z2_new], dim=-1)

        log_det_jacobian = torch.sum(s, dim=-1)

        return z_new, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows):
        super().__init__()
        self.flows = nn.ModuleList([RealNVP(latent_dim=latent_dim, hidden_dim=latent_dim) for _ in range(num_flows)])

    def _chunk_forward(z):
        D = z.shape[-1]
        d = D // 2
        return z[..., :d], z[..., d:]

    def _chunk_inverse(z):
        D = z.shape[-1]
        d = D // 2
        return z[..., :D - d], z[..., D - d:]

    def forward(self, z):
        log_det_jacobian = 0.0
        for flow in self.flows:
            z, ldj = flow(z)
            z1, z2 = NormalizingFlow._chunk_forward(z)
            z = torch.cat([z2, z1], dim=-1)
            log_det_jacobian += ldj
        return z, log_det_jacobian

    def inverse(self, z):
        for flow in reversed(self.flows):
            z2, z1 = NormalizingFlow._chunk_inverse(z)
            s = flow.scale_net(z1)
            t = flow.translate_net(z1)
            z2 = (z2 - t) * torch.exp(-s)
            z = torch.cat([z1, z2], dim=-1)
        return z
