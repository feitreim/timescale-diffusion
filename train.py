from toml.decoder import TomlDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import List, Tuple
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import toml
import argparse

from blocks.unet import UNet
from data.dali_loader import DALIDataset
from data.utils import unpack

# Args
parser = argparse.ArgumentParser(description="train the timescale diffusion model")
parser.add_argument("config_file", help="Path to the configuration file")
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

# Model(s)
# Just UNET for now
model_unopt = UNet(**config["model"])
model = torch.compile(model_unopt, **config["compile"])

# optim
optimizer = optim.AdamW(model_unopt.parameters(), lr=learning_rate)

# dataset
dataset = DALIDataset(**config["data"])

for batch in tqdm(dataset):
    x, y, t = unpack(batch)
    x_hat = model(x, t)

    loss = torch.pow((y - x_hat), 2.0).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
