import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        activation = nn.Tanh()
        block = []
        block.append(nn.Conv2d(in_dim, h_dim, kernel_size, padding=padding))
        block.append(activation)
        block.append(nn.Conv2d(h_dim, h_dim, kernel_size, padding=padding))
        block.append(activation)
        block.append(nn.Conv2d(h_dim, in_dim, kernel_size, padding=padding))
        block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class ResDown(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        self.block = ResBlock(in_dim, h_dim, kernel_size, padding)
        self.down = nn.Conv2d(in_dim, h_dim, kernel_size=kernel_size, stride=2, padding=padding)
        self.act = nn.Tanh()

    def forward(self, x, t):
        x = self.block(x)
        x = self.down(x)
        x = self.act(x)
        return x


class ResUp(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size, padding):
        super().__init__()
        self.in_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size, padding=padding)
        self.block = ResBlock(in_dim // 2, h_dim, kernel_size, padding)
        self.up = nn.ConvTranspose2d(in_dim // 2, h_dim, kernel_size=2, stride=2)
        self.act = nn.Tanh()

    def forward(self, x, t):
        x = self.in_conv(x)
        x = self.act(x)
        x = self.block(x)
        x = self.up(x)
        x = self.act(x)
        return x
