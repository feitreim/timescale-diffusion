import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor

from blocks.resnet import ResBlock


class AttnBlock(nn.Module):
    """
    Attention Block for Diffusion Model
    Cross-Attends the timestamp encoding (t) with inputs (x)
    Args
        -- e_dim (int): embedding dimension, inputs will be reshaped to [B, S, e_dim]
    """

    def __init__(self, e_dim):
        super().__init__()
        self.q_proj = torch.nn.Parameter(
            torch.randn(1, e_dim, e_dim), requires_grad=True
        )
        self.kv_proj = torch.nn.Parameter(
            torch.randn(1, e_dim, e_dim), requires_grad=True
        )
        self.attn_proj = torch.nn.Parameter(
            torch.randn(1, e_dim, e_dim), requires_grad=True
        )
        self.e_dim = e_dim
        self.norm = nn.LayerNorm(e_dim)

    """
    input shapes:
        -- x => (B, C, H, W)
        -- t => (B, e_dim)
    """

    def forward(self, x: Tensor, t: Tensor):
        shape = x.shape
        x = x.view(shape[0], -1, self.e_dim)
        x = x @ self.kv_proj
        x_n = self.norm(x)
        t = t.unsqueeze(1).expand(-1, x.shape[1], -1)
        t = t @ self.q_proj
        t = self.norm(t)
        # let the output of sdpa be the residual (r) for x
        r = f.scaled_dot_product_attention(t, x_n, x_n) @ self.attn_proj
        x = x + nn.functional.tanh(r)
        return x.view(*shape)


class AttnDown(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, padding, e_dim, heads):
        super().__init__()
        self.in_conv = nn.Sequential(
            ResBlock(in_dims, in_dims, kernel_size, padding),
            nn.Conv2d(in_dims, in_dims, kernel_size, padding=padding),
            nn.Tanh(),
        )
        attn = [AttnBlock(e_dim) for i in range(heads)]
        self.attention = nn.ModuleList(attn)
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size, stride=2, padding=padding),
            nn.Tanh(),
            ResBlock(out_dims, out_dims, kernel_size, padding),
        )

    def forward(self, x, t):
        x = self.in_conv(x)
        for attn in self.attention:
            x = attn(x, t)
        x = self.down_conv(x)
        return x


class AttnUp(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, padding, e_dim, heads):
        super().__init__()
        self.in_conv = nn.Sequential(
            ResBlock(in_dims, in_dims, kernel_size, padding),
            nn.Conv2d(in_dims, in_dims // 2, kernel_size, padding=padding),
            nn.Tanh(),
        )
        attn = [AttnBlock(e_dim) for i in range(heads)]
        self.attention = nn.ModuleList(attn)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_dims // 2, out_dims, kernel_size=2, stride=2),
            nn.Tanh(),
            ResBlock(out_dims, out_dims, kernel_size, padding),
        )

    def forward(self, x, t):
        x = self.in_conv(x)
        for attn in self.attention:
            x = attn(x, t)
        x = self.up_conv(x)
        return x
