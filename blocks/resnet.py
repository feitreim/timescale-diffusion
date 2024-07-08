import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class ResBlock(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        activation = nn.SiLU()
        block = []
        block.append(nn.Conv2d(in_dim, h_dim, kernel_size, padding=padding))
        block.append(activation)
        block.append(nn.Conv2d(h_dim, h_dim, kernel_size, padding=padding))
        block.append(activation)
        block.append(nn.Conv2d(h_dim, in_dim, kernel_size, padding=padding))
        block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = x + self.block(x)
        return x


class ResDown(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        self.block = ResBlock(in_dim, h_dim, kernel_size, padding)
        self.down = nn.Conv2d(
            in_dim, h_dim, kernel_size=kernel_size, stride=2, padding=padding
        )
        self.act = nn.SiLU()

    def forward(self, x, t):
        x = self.block(x)
        x = self.down(x)
        x = self.act(x)
        return x


class ResUp(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        self.block = ResBlock(in_dim, h_dim, kernel_size, padding)
        self.up = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=2, stride=2)
        self.act = nn.SiLU()

    def forward(self, x, t):
        x = self.block(x)
        x = self.up(x)
        x = self.act(x)
        return x
