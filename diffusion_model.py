import torch
import torch.nn as nn
from typing import Dict

from blocks.unet import UNet
from vqvae.model import VQVAE

class LTDM(nn.Module):
    def __init__(self, unet_config: Dict, vqvae_config: Dict):
        super().__init__()
        self.unet = UNet(**unet_config) 
        self.vae = VQVAE(**vqvae_config)
    
    def forward(self, x, t):
        z = self.generate_latent(x)
        z_hat = self.unet(z, t)
        embed_loss, x_hat, perplexity, _ = self.vae.generate_output_from_latent(z_hat)
        return embed_loss, x_hat, perplexity

    def generate_latent(self, x):
        return self.vae.generate_latent(x)

    def diffusion_step(self, z, t):
        return self.unet(z, t)
    
    def generate_output_from_latent(self, z_hat):
        return self.vae.generate_output_from_latent(z_hat)

