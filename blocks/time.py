import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from blocks.attention import AttnBlock


class TimeEmbedding1D(nn.Module):
    """
    Embed a timestamp of t_dim into e_dim's
    """

    def __init__(self, t_dim, e_dim):
        block = []
        block.append(nn.Linear(t_dim, e_dim, bias=True))
        block.append(nn.SiLU)
        block.append(nn.Linear(e_dim, e_dim, bias=True))
        self.block = nn.Sequential(*block)

    def forward(self, t):
        t = self.block(t.squeeze())
        return t