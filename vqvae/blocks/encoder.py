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
    - stacks: number of encoder stackers, 1 for C, 64, 64, 2 for C, 16, 16 (from 256)

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stacks):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        conv_stack = []
        conv_stack += [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        ]
        if stacks == 2:
            conv_stack += [
                nn.Conv2d(
                    h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
                ),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            ]
        self.conv_stack = nn.Sequential(*conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64, 1)
    encoder_out = encoder(x)
    print("Encoder out shape:", encoder_out.shape)