import torch
import torch.nn as nn
from typing import Dict

from blocks.unet import UNet
from vqvae.model import VQVAE

"""
                        training pass
         ┌─────────┐                      ┌─────────┐   ┌─────┐
         │         │  ┌────────────────┐  │         ├───▶x_hat│
┌─────┐  │  VQVAE  │  │      UNET      │  │  VQVAE  │   └─────┘
│  x  │──▶         │─┬▶                ├┬─▶         │
└─────┘  │ Encoder │ ││ Diffusion Step ││ │ Decoder │   ┌─────┐
         │         │ │└────────────────┘│ │         ├───▶y_hat│
         └─────────┘ │                  │ └─────────┘   └─────┘
                     └───no diffusion───┘
"""


class LTDM(nn.Module):
    def __init__(self, unet_config: Dict, vqvae_config: Dict):
        """
        Latent Time Diffusion Model
        for some image x, encode to latent z, diffuse through time to z_hat
        and decode to image y_hat
        Args:
            -- unet_config: see blocks.unet for more info
                -- depth: int
                -- down_layers: List[str]
                -- up_layers: List[str]
                -- in_dims: int
                -- h_dims: int
                -- out_dims: int
                -- kernel_size: int
                -- padding: int
                -- e_dims: int
            -- vqvae_config: see vqvae.model for more info
                -- h_dim: int
                -- res_h_dim: int
                -- n_res_layers: int
                -- n_embeddings: int
                -- embedding_dim: int
                -- beta: int
        """
        super().__init__()
        self.unet = UNet(**unet_config)
        self.vae = VQVAE(**vqvae_config)

    def forward(self, x, t):
        z = self.generate_latent(x)
        z_hat = self.unet(z, t)
        embed_loss_y, y_hat, perp_y, _ = self.generate_output_from_latent(z_hat)
        embed_loss_x, x_hat, perp_x, _ = self.generate_output_from_latent(z)
        return (embed_loss_x, embed_loss_y, x_hat, y_hat, perp_x, perp_y)

    def generate_latent(self, x):
        return self.vae.generate_latent(x)

    def generate_output_from_latent(self, z_hat):
        return self.vae.generate_output_from_latent(z_hat)

    def diffusion_step(self, x, t):
        z = self.generate_latent(x)
        z_hat = self.unet(z, t)
        _, x_hat, _, _ = self.generate_output_from_latent(z_hat)
        return x_hat

# ----------- Other Related Functions ---------------
from utils import convert_timestamp_to_periodic, download_artifact
import toml

def load_model_from_artifact(artifact):
    state_dict_path, arg_dict_path = download_artifact(artifact)
    config = toml.load(arg_dict_path)
    model = LTDM(config["unet_config"], config["vqvae_config"])
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model

def non_ar_video(model, frame, start, end, step):
    s_time = convert_timestamp_to_periodic(start)
    frame = frame.squeeze().reshape(1, 3, 256, 256)
    frames = []
    for i in range(start, end, step):
        c_time = convert_timestamp_to_periodic(i)
        t = torch.stack([s_time, c_time]).unsqueeze(0)
        
        frames.append(model.diffusion_step(frame, t))
    return frames
        





















