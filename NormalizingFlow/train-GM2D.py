import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import NormalizingFlow

scales = torch.tensor([
    1.0,
    1.0,
])

means = torch.tensor([
    [4, 6],
    [-5, -3],
])

def sample_date(n):
    assert n % len(scales) == 0
    z = torch.tensor([])
    for i in range(len(scales)):
        z = torch.cat([z, torch.randn(n // len(scales), 2) * scales[i] + torch.tensor([means[i][0], means[i][1]])], dim=0)
    return z

def train():
    # Hyperparameters
    latent_dim = 2
    num_flows  = 6
    batch_size = 128
    epochs     = 2000
    learning_rate = 1e-4
    sample_folder = '../samples/NormalizingFlow/GM2D'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    data = sample_date(2**14)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    os.makedirs(sample_folder, exist_ok=True)

    # Model, optimizer
    model = NormalizingFlow(latent_dim, num_flows)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            z0 = batch.to(device).float()
            zK, log_det_jacobian = model(z0)
            log_prob_zK = -0.5 * torch.sum(zK**2, dim=1) - (latent_dim / 2) * torch.log(torch.tensor(2 * torch.pi))
            loss = -torch.mean(log_prob_zK + log_det_jacobian)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            with torch.no_grad():
                z_sample = torch.randn(8192, latent_dim)
                x_sample = model.inverse(z_sample)

                H, _, _ = np.histogram2d(x_sample[:, 0], x_sample[:, 1], bins=100, range=[[-10, 10], [-10, 10]], density=True)

                plt.figure(figsize=(6, 6))
                plt.imshow(H.T, origin='lower', extent=[-10, 10, -10, 10], cmap='plasma', aspect='auto')
                plt.colorbar()

                plt.scatter(means[:, 0], means[:, 1], color='black', s=50, marker='o', label='centroids', alpha=0.4)

                plt.xlim(-10, 10)
                plt.ylim(-10, 10)
                plt.title(f'Sample at Epoch {epoch+1}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.savefig(os.path.join(sample_folder, f'sample_epoch_{epoch+1}.png'))
                plt.close()

if __name__ == '__main__':
    train()
