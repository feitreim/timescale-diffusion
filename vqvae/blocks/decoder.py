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
    - stacks: number of up/down convs.

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stacks):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        inverse_conv_stack = build_inv_conv_stack(
            stacks, in_dim, h_dim, kernel, stride, res_h_dim, n_res_layers
        )
        self.inverse_conv_stack = nn.Sequential(*inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


def build_inv_conv_stack(
    stacks, in_dim, h_dim, kernel, stride, res_h_dim, n_res_layers
):
    conv_stack = []
    conv_stack += [
        nn.ConvTranspose2d(
            in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
        ),
        ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        nn.ConvTranspose2d(
            h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
        ),
        nn.ReLU(True),
    ]
    if stacks >= 2:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim // 2,
                h_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(True),
        ]
    else:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim // 2,
                3,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
            ),
            nn.Sigmoid(),
        ]
        return conv_stack
    if stacks >= 3:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
        ]
    else:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim, 3, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            nn.Sigmoid(),
        ]
        return conv_stack
    if stacks >= 4:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.Sigmoid(),
        ]
    else:
        conv_stack += [
            nn.ConvTranspose2d(
                h_dim // 2, 3, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            nn.Sigmoid(),
        ]
    return conv_stack


def test():
    # random data
    x = np.random.random_sample((16, 128, 16, 16))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(128, 128, 3, 64, 1)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = Decoder(128, 128, 3, 64, 2)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = Decoder(128, 128, 3, 64, 3)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = Decoder(128, 128, 3, 64, 4)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)


if __name__ == "__main__":
    test()
