import torch.nn as nn
from vqvae.models.encoder import Encoder
from vqvae.models.quantizer import VectorQuantizer
from vqvae.models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        stacks,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, stacks)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x):
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity, z_e

    def generate_latent(self, x):
        """
        First Half of the forward pass, returns just a latent embedding.
        """
        z_e = self.encoder(x)

        latent = self.pre_quantization_conv(z_e)
        return latent

    def generate_output_from_latent(self, latent):
        """
        Generate the output image from a continous latent space.
        """
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(latent)

        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity, latent
