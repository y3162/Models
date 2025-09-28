import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import NormalizingFlow

scales = torch.tensor([
    1.0,
    1.0,
])

means = torch.tensor([
    [2, 3, 4],
    [-4, -4, -1],
])

def sample_date(n):
    assert n % len(scales) == 0
    z = torch.tensor([])
    for i in range(len(scales)):
        z = torch.cat([z, torch.randn(n // len(scales), 3) * scales[i] + torch.tensor([means[i][0], means[i][1], means[i][2]])], dim=0)
    return z

def train():
    # Hyperparameters
    latent_dim = 3
    num_flows  = 6
    batch_size = 128
    epochs     = 2000
    learning_rate = 1e-4
    sample_folder = '../samples/NormalizingFlow/GM3D'

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

                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter( x_sample[:, 0], x_sample[:, 1], x_sample[:, 2], alpha=0.1, s=2, c='blue')
                ax.scatter(means[:, 0], means[:, 1], means[:, 2], color='red', s=50, marker='o', label='centroids')
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.set_zlim(-10, 10)
                ax.set_title(f'Sample at Epoch {epoch+1}')
                ax.legend()
                plt.savefig(os.path.join(sample_folder, f'sample_epoch_{epoch+1}.png'))
                plt.close()

if __name__ == '__main__':
    train()
