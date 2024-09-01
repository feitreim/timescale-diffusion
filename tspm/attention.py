import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.init import xavier_uniform_ as _xav
from torch import Tensor

from tspm.resnet import ResBlock


class AttnBlock(nn.Module):
    """
    Attention Block for Diffusion Model
    Cross-Attends the timestamp encoding (t) with inputs (x)
    Args
        -- e_dim (int): embedding dimension, inputs will be reshaped to [B, S, e_dim]
    """

    def __init__(self, e_dim):
        super().__init__()
        self.q_proj = torch.nn.Parameter(_xav(torch.randn(1, e_dim, e_dim)), requires_grad=True)
        self.kv_proj = torch.nn.Parameter(_xav(torch.randn(1, e_dim, e_dim * 2)), requires_grad=True)
        self.attn_proj = torch.nn.Parameter(_xav(torch.randn(1, e_dim, e_dim)), requires_grad=True)
        self.mlp_up = torch.nn.Parameter(_xav(torch.randn(1, e_dim, e_dim * 2)), requires_grad=True)
        # S = 256 hard coded rn
        self.mlp_bias = torch.nn.Parameter(_xav(torch.randn(1, e_dim * 2, e_dim * 2)), requires_grad=True)
        self.mlp_down = torch.nn.Parameter(_xav(torch.randn(1, e_dim * 2, e_dim)), requires_grad=True)

        self.e_dim = e_dim
        self.norm = nn.LayerNorm(e_dim)

    def forward(self, x: Tensor, t: Tensor):
        """
        input shapes:
            -- x => (B, C, H, W)
            -- t => (B, e_dim)
        """
        B, S, shape = x.shape[0], x.shape[1], x.shape
        x = x.reshape(B, -1, self.e_dim)
        t = t.unsqueeze(1).expand(-1, S, -1)
        q = self.norm(x @ self.q_proj)
        # do both k and v at the same time in one mm to speed up
        # in self attn you can actually do all 3 at once
        kv = t @ self.kv_proj
        k, v = torch.split(kv, self.e_dim, dim=-1)
        k, v = self.norm(k), self.norm(v)
        # take the attention map (phi) and add it to (x)
        r = f.scaled_dot_product_attention(q, k, v) @ self.attn_proj
        r = f.gelu((r @ self.mlp_up) + self.mlp_bias)
        x = x + (r @ self.mlp_down)
        return x.reshape(*shape).to(memory_format=torch.channels_last)


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
