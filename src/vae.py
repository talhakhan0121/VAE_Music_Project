from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims=(256, 128)):
        super().__init__()
        h1, h2 = hidden_dims

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, h1)
        self.enc_fc2 = nn.Linear(h1, h2)
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, h2)
        self.dec_fc2 = nn.Linear(h2, h1)
        self.out = nn.Linear(h1, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        x_hat = self.out(h)  # reconstruction
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()
