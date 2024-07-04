import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class UNet(nn.Module):
    def __init__(
        self,
        in_dims: int,
        h_dims: int,
        out_dims: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()
        dims = [h_dims // 8, h_dims // 4, h_dims//2, h_dims]
        self.in_conv = nn.Conv2d(in_dims, dims[0] // 2, kernel_size, padding=padding)
        down_layers = [
            nn.Sequential(
                nn.Conv2d(h // 2, h // 2, kernel_size, padding=padding),
                nn.SiLU(),
                nn.Conv2d(h // 2, h, kernel_size, padding=padding),
                nn.SiLU(),
            ) for h in dims
        ] # list comprehension, make the sequential block for each value in dims.
        self.down_layers = nn.ModuleList(down_layers)
        up_layers = [
            nn.Sequential(
                nn.Conv2d(h, h // 2, kernel_size, padding=padding),
                nn.SiLU(),
                nn.Conv2d(h // 2, h // 2, kernel_size, padding=padding),
                nn.SiLU(),
            ) for h in reversed(dims)
        ] # list comprehension, make the sequential block for each value in dims.
        self.up_layers = nn.ModuleList(up_layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)
        self.out_conv = nn.Conv2d(dims[0]//2, out_dims, kernel_size, padding=padding)


    def forward(self, x):
        x = self.in_conv(x)
        x_1 = self.down_layers[0](x)
        x_2 = self.pool(x_1)
        x_2 = self.down_layers[1](x_2)
        x_3 = self.pool(x_2)
        x_3 = self.down_layers[2](x_3)
        x_4 = self.pool(x_3)
        x_4 = self.down_layers[3](x_4)
        x_up = self.up_layers[0](x_4)
        x_up = self.up(x_up) + x_3
        x_up = self.up_layers[1](x_up)
        x_up = self.up(x_up) + x_2
        x_up = self.up_layers[2](x_up)
        x_up = self.up(x_up) + x_1
        x_up = self.up_layers[3](x_up)
        x_up = self.out_conv(x_up)
        return x_up
