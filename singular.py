"""
An experiment on the effect of increasing a singular value inorder to move through time
inside of the latent space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from torchinfo import summary
from typing import List, Tuple
from tqdm import tqdm
import toml
import argparse
import wandb
import pathlib
import random
import os
from pathlib import Path
from multiprocessing import freeze_support

from vqvae.model import VQVAE
from data.pair_dali import PairDataset

# from data.frame_random_index import FrameDataset
from utils import unpack, download_artifact, convert_timestamp_to_periodic
from losses.reconstructionLosses import MixReconstructionLoss

# -------------- Functions


def save_model(model):
    artifact = wandb.Artifact(args.name, type="model")
    artifact.add_file(local_path=args.config_file, name="model_config", is_tmp=True)
    checkpoint_path = pathlib.Path(f"./.checkpoints") / f"{args.name}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{str(checkpoint_path.resolve())}/model_state_dict.pth"
    torch.save(model.state_dict(), checkpoint_path)
    artifact.add_file(local_path=checkpoint_path, name="model_state_dict", is_tmp=True)
    wandb.log_artifact(artifact)


def training_step(batch_idx, batch):
    x, y, t = unpack(batch, device)
    t = t.flatten(start_dim=1).unsqueeze_(1)

    z_x = model.generate_latent(x)
    u, s, vh = torch.linalg.svd(z_x)
    s = pusher(s, t)
    z_y = u @ torch.diag_embed(s) @ vh
    embed_loss_y, y_hat, perplexity_y, _ = model.generate_output_from_latent(z_y)
    embed_loss_x, x_hat, perplexity_x, _ = model.generate_output_from_latent(z_x)

    pred_loss = ssim_loss(y_hat, y)
    orig_loss = ssim_loss(x_hat, x)
    loss = pred_loss + orig_loss + embed_loss_y + embed_loss_x

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % logging_rate == 0:
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/pred_loss": pred_loss.item(),
                "train/orig_loss": orig_loss.item(),
                "train/embed_loss_y": embed_loss_y.item(),
                "train/embed_loss_x": embed_loss_x.item(),
                "train/perplexity_y": perplexity_y.item(),
                "train/perplexity_x": perplexity_x.item(),
            }
        )

    if batch_idx % (logging_rate**2) == 0:
        caption = "left: input, middle left: input recon, middle right: target recon, right: target"
        mosaic = torch.cat([x[:4], x_hat[:4], y_hat[:4], y[:4]], dim=-1)
        wandb.log(
            {"train/images": [wandb.Image(img, caption=caption) for img in mosaic]}
        )


@torch.no_grad
def validation_step(batch_idx, batch):
    x, y, t = unpack(batch, device)
    x_hat = model(x, t)

    loss = ssim_loss(x_hat, y)

    if batch_idx % 10 == 0:
        wandb.log({"val/loss": loss.item()})


@torch.no_grad
def running_average_weights(model: nn.Module, path, beta):
    state = torch.load(path).state_dict()
    for name, param in model.named_parameters():
        param.data = (param.data * beta) + (state[name].data * (1 - beta))
    torch.save(model, path)


# --------------- Script

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description="train the timescale diffusion model")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("--name", help="run name.")
    args = parser.parse_args()

    # Load config
    config = toml.decoder.load(args.config_file)

    # dataset
    batch_size = config["data"]["batch_size"]
    dataset = PairDataset(**config["data"])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # Hyperparameters
    learning_rate = config["hp"]["lr"] if "lr" in config["hp"] else 0.001
    num_epochs = config["hp"]["num_epochs"] if "num_epochs" in config["hp"] else 5
    logging_rate = (
        config["hp"]["logging_rate"] if "logging_rate" in config["hp"] else 50
    )
    epoch_size = config["hp"]["epoch_size"] if "epoch_size" in config["hp"] else 100000

    # Model(s)
    # Just UNET for now
    model_unopt = VQVAE(**config["model"])
    summary(model_unopt, input_size=(batch_size, 3, 256, 256))
    model_unopt = model_unopt.to(device)
    model = torch.compile(model_unopt, **config["compile"])

    class Pusher(nn.Module):
        def __init__(self, last_dim_size):
            super().__init__()
            bias = torch.randn(1, 1, last_dim_size)
            weight = torch.randn(1, 14, last_dim_size)
            self.bias = torch.nn.Parameter(bias, requires_grad=True)
            self.weight = torch.nn.Parameter(weight, requires_grad=True)

        def forward(self, x, t):
            """
            inputs:
                x <- singular values [b d]
                t <- start and end timestamps [b 14]

            outputs:
                x' <- singular values [b d]
            """
            return x + (self.bias + (t @ self.weight))

    pusher = Pusher(16).to(device)

    # optim
    params = list(model_unopt.parameters())
    params += list(pusher.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate)

    # ema of weights
    ema_id = random.randint(0, 2000000)
    ema_path = f"./.running_avgs/{ema_id}/"
    ema_beta = config["ema"]["beta"]
    ema_interval = config["ema"]["interval"]
    Path(ema_path).mkdir(parents=True, exist_ok=True)
    ema_path = Path(ema_path) / "lastweight.ckpt"
    torch.save(model_unopt, ema_path)

    # loss
    ssim_loss = MixReconstructionLoss()

    # dataset
    # val_dataset = FrameDataset(**config['val_data'])
    # wandb
    wandb.init(project="latent-rotation", name=args.name)
    save_model(model)

    for e in range(num_epochs):
        wandb.log({"epoch": e})
        for batch_idx, batch in tqdm(enumerate(dataset)):
            training_step(batch_idx, batch)
            if batch_idx % ema_interval == 0:
                running_average_weights(model_unopt, ema_path, ema_beta)
            if batch_idx >= epoch_size:
                save_model(model)
                break
