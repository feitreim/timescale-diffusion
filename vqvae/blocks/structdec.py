import torch
import torch.nn as nn
import numpy as np
from vqvae.blocks.residual import ResidualStack
from tspm.time import TimeEmbedding2D


class TimeBias(nn.Module):
    def __init__(self, t_dim, dim_size):
        super().__init__()
        self.bias = nn.Parameter(
            torch.rand((1, 1, dim_size, dim_size)), requires_grad=True
        )
        self.scalar = nn.Parameter(
            torch.randn((1, t_dim * 2, dim_size * dim_size)), requires_grad=True
        )
        self.spatial = dim_size  # resolution

    def forward(self, x, t):
        B = x.shape[0]
        scalar = t @ self.scalar
        scalar = scalar.view(B, 1, self.spatial, self.spatial)
        bias = self.bias * scalar
        return torch.cat([x, bias], dim=1)


class StructuralDecoder(nn.Module):
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
        super(StructuralDecoder, self).__init__()
        kernel = 4
        stride = 2

        self.first_time_embed = TimeBias(5, 32)
        self.first_up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim + 1, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            nn.ReLU(True),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(True),
        )
        self.second_time_embed = TimeBias(5, 64)
        self.second_up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                (h_dim // 2) + 1,
                h_dim // 2,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
            ),
            nn.ReLU(True),
            ResidualStack(h_dim // 2, h_dim // 2, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                (h_dim // 2),
                h_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(True),
        )
        self.third_time_embed = TimeBias(5, 128)
        self.third_up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                h_dim + 1, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                h_dim // 2, 3, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        t = t.flatten(1)
        x = self.first_time_embed(x, t)
        x = self.first_up_conv(x)

        x = self.second_time_embed(x, t)
        x = self.second_up_conv(x)

        x = self.third_time_embed(x, t)
        x = self.third_up_conv(x)

        return x


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
    decoder = StructuralDecoder(128, 128, 3, 64, 1)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = StructuralDecoder(128, 128, 3, 64, 2)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = StructuralDecoder(128, 128, 3, 64, 3)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
    # test decoder
    decoder = StructuralDecoder(128, 128, 3, 64, 4)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)


if __name__ == "__main__":
    test()
