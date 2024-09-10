import torch
import torch.nn as nn
import numpy as np
from vqvae.blocks.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    - stacks: number of up/down convs.

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stacks):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2

        conv_stack = build_conv_stack(stacks, in_dim, h_dim, kernel, stride, res_h_dim, n_res_layers)

        self.conv_stack = nn.Sequential(*conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


def build_conv_stack(stacks, in_dim, h_dim, kernel, stride, res_h_dim, n_res_layers):
    conv_stack = []
    conv_stack += [
        nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
        nn.BatchNorm2d(h_dim // 2),
        nn.LeakyReLU(True),
    ]
    if stacks >= 2:
        conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(True),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        ]
    else:
        conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(True),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        ]
        return conv_stack
    if stacks >= 3:
        conv_stack += [nn.Conv2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1), nn.BatchNorm2d(h_dim // 2), nn.LeakyReLU(True)]
    else:
        return conv_stack
    if stacks >= 4:
        conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(True),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        ]
    else:
        conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(True),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        ]
    return conv_stack


def test():
    # random data
    x = np.random.random_sample((16, 3, 256, 256))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(3, 128, 3, 64, 1)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    # test encoder
    encoder = Encoder(3, 128, 3, 64, 2)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    # test encoder
    encoder = Encoder(3, 128, 3, 64, 3)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    # test encoder
    encoder = Encoder(3, 128, 3, 64, 4)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)


if __name__ == '__main__':
    test()
