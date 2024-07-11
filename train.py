import torch
import torch.nn as nn
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

from blocks.unet import UNet
from data.pair_dali import PairDataset

# from data.frame_random_index import FrameDataset
from utils import unpack
from losses.reconstructionLosses import MixReconstructionLoss

# -------------- Functions


def save_model(model, path):
    torch.save(model.state_dict(), path)
    artifact.add_file(local_path=path, name="model_state_dict", is_tmp=True)
    wandb.log_artifact(artifact)


def training_step(batch_idx, batch):
    x, y, t = unpack(batch, device)
    x_hat = model(x, t)

    loss = ssim_loss(x_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % logging_rate == 0:
        wandb.log({"train/loss": loss.item()})

    if batch_idx % (logging_rate**2) == 0:
        caption = "left: input, middle: prediction, right: target"
        mosaic = torch.cat([x[:4], x_hat[:4], y[:4]], dim=-1)
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


# --------------- Script

# Args
parser = argparse.ArgumentParser(description="train the timescale diffusion model")
parser.add_argument("config_file", help="Path to the configuration file")
parser.add_argument("--name", help="run name.")
args = parser.parse_args()

# Load config
config = toml.decoder.load(args.config_file)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# Hyperparameters
batch_size = config["data"]["batch_size"]
learning_rate = config["hp"]["lr"] if "lr" in config["hp"] else 0.001
num_epochs = config["hp"]["num_epochs"] if "num_epochs" in config["hp"] else 5
logging_rate = config["hp"]["logging_rate"] if "logging_rate" in config["hp"] else 50
epoch_size = config["hp"]["epoch_size"] if "epoch_size" in config["hp"] else 100000

# Model(s)
# Just UNET for now
model_unopt = UNet(**config["model"])
summary(model_unopt, input_size=((batch_size, 3, 256, 256), (batch_size, 2, 7)))
model_unopt = model_unopt.to(device)
model = torch.compile(model_unopt, **config["compile"])

# optim
optimizer = optim.AdamW(model_unopt.parameters(), lr=learning_rate)

# loss
ssim_loss = MixReconstructionLoss()

# dataset
dataset = DALIDataset(**config["data"])
# val_dataset = FrameDataset(**config['val_data'])
# wandb
wandb.init(project="timescale-diffusion", name=args.name)
artifact = wandb.Artifact(args.name, type="model")
artifact.add_file(local_path=args.config_file, name="model_config", is_tmp=True)
checkpoint_path = pathlib.Path(f"./.checkpoints") / f"{args.name}"
checkpoint_path.mkdir(parents=True, exist_ok=True)
checkpoint_path = f"{str(checkpoint_path.resolve())}/model_state_dict.pth"
save_model(model, checkpoint_path)

for e in range(num_epochs):
    wandb.log({"epoch": e})
    for batch_idx, batch in tqdm(enumerate(dataset)):
        training_step(batch_idx, batch)
        if batch_idx >= epoch_size:
            save_model(model, checkpoint_path)
            break
