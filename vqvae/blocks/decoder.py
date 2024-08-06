import torch
import torch.nn as nn
import numpy as np
from vqvae.blocks.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    - stacks: number of decoder stackers, 1 for C, 64, 64, 2 for C, 16, 16 (from 256)

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stacks):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        stack_1_out_dim = 3
        if stacks > 1:
            stack_1_out_dim = h_dim

        inverse_conv_stack = []
        inverse_conv_stack += [
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                h_dim // 2,
                stack_1_out_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
        ]
        if stacks >= 2:
            inverse_conv_stack += [
                nn.ConvTranspose2d(
                    h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
                ),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                nn.ConvTranspose2d(
                    h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1
                ),
            ]

        self.inverse_conv_stack = nn.Sequential(*inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print("Dncoder out shape:", decoder_out.shape)
