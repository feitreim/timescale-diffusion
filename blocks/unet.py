import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from blocks.attention import AttnUp, AttnDown, AttnBlock
from blocks.resnet import ResUp, ResDown
from blocks.time import TimeEmbedding1D


class UNet(nn.Module):
    """
    UNet Diffusion Block
    Args:
        -- depth: how many layers
        -- down_layers: what kind of down layer at each step, 1 less than depth
        -- up_layers: what kind of up layer at each step, 1 less than depth
        -- in_dims: input channel dimension
        -- h_dims: max hidden dims
        -- out_dims: output channel dimension
        -- kernel_size: conv kernel size
        -- padding: conv padding
        -- e_dims: embedding dimension for the timestamp and attention blocks.
    """

    def __init__(
        self,
        depth: int,
        down_layers: List[str],
        up_layers: List[str],
        in_dims: int,
        h_dims: int,
        out_dims: int,
        kernel_size: int,
        padding: int,
        e_dims: int,
    ):
        super().__init__()
        dims = [h_dims // (2**i) for i in range(0, depth).__reversed__()]
        print(dims)
        activation = nn.SiLU()
        self.t_emb = TimeEmbedding1D(7, e_dims)
        self.in_conv = nn.Conv2d(in_dims, dims[0] // 2, kernel_size, padding=padding)
        down = [
            get_layer(down_layers[i], h // 2, h, kernel_size, padding, e_dims)
            for i, h in enumerate(dims[:-1])
        ]
        up = [
            nn.Sequential(
                nn.Conv2d(h * 2, h, kernel_size, padding=padding),
                activation,
                get_layer(up_layers[i], h, h // 2, kernel_size, padding, e_dims),
            )
            for i, h in enumerate(reversed(dims[:-1]))
        ]

        self.down_layers = nn.ModuleList(down)
        self.bottom_layer = UNetBottom(h_dims, kernel_size, padding, e_dims)
        self.up_layers = nn.ModuleList(up)
        self.out_conv = nn.Conv2d(dims[0] // 2, out_dims, kernel_size, padding=padding)
        self.n_ups = len(self.up_layers) - 1

    def forward(self, x, t):
        x = self.in_conv(x)
        t = self.t_emb(t)
        states = []
        for i, l in enumerate(self.down_layers):
            states.append(x := l(x, t))
        x = self.bottom_layer(x, t)
        for i, l in enumerate(self.up_layers):
            x = l(torch.cat([x, states[self.n_ups - i]], dim=1))
        x = self.out_conv(x)
        return x


class UNetBottom(nn.Module):
    def __init__(self, h_dims, kernel_size, padding, e_dims):
        super().__init__()
        self.in_conv = nn.Conv2d(h_dims // 2, h_dims, kernel_size, padding=padding)
        self.attn = AttnBlock(e_dims)
        self.out_conv = nn.Conv2d(h_dims, h_dims // 2, kernel_size, padding=padding)

    def forward(self, x, t):
        x = self.in_conv(x)
        x = self.attn(x, t)
        x = self.out_conv(x)
        return x


def get_layer(name: str, in_dim, out_dim, kernel_size, padding, e_dim) -> nn.Module:
    if name == "ResDown":
        return ResDown(in_dim, out_dim, kernel_size, padding)
    if name == "AttnDown":
        return AttnDown(in_dim, out_dim, kernel_size, padding, e_dim)
    if name == "ResUp":
        return ResUp(in_dim, out_dim, kernel_size, padding)
    # if name == 'AttnUp':
    return AttnUp(in_dim, out_dim, kernel_size, padding, e_dim)
