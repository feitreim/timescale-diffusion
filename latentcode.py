import argparse
from dataclasses import dataclass
import random
from pathlib import Path

from numpy import require
import toml
import gc
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import xavier_uniform_ as _xav
import wandb
import schedulefree
from torchinfo import summary
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from tqdm import tqdm
from itertools import zip_longest, chain, islice
from typing import List
from fvcore.nn.precise_bn import update_bn_stats

from data.pair_dali import PairDataset
from losses.recon import MixReconstructionLoss
from utils import unpack, convert_timestamp_to_periodic_vec
from vqvae.model import VQVAE
from vqvae.blocks.encoder import Encoder
from vqvae.blocks.decoder import Decoder
from vqvae.blocks.quantizer import VectorQuantizer
from vqvae.blocks.residual import ResidualStack
from tspm.attention import AttnBlock
from tspm.time import TimeEmbedding2D

"""
current training:

┌───┐         ┌─────┐   ┌───┐      ┌─────┐    ┌───┐  
│ X ├─────────▶Outer├───▶s_x├──────▶Outer├────▶ x │  
├───┤         │Model│   ├───┤      │Model│    ├───┤  
│ Y ├─────────▶ Enc ├───▶s_y├──────▶ Dec ├────▶ y │  
└───┘         └─────┘   └───┘      └─────┘    └───┘  
           ┌─────┐                    ┌─────┐        
┌───┐      │Inner│  ┌───┐cross ┌───┐  │Inner│   ┌───┐
│s_x├──────▶Model├──▶z_x├──────▶z_y├──▶Model├───▶s_y│
└───┘      │ Enc │  └───┘attn. └───┘  │ Dec │   └───┘
           └─────┘                    └─────┘        
"""


# -------------- Model
class TSVAE(nn.Module):
    """
    we define two latent spaces, the structural latent space (s)
    and the temporal latent space (z). we then use subscript to
    indicate the image it represents, for example (s_x) is the
    structural latent space of image x.
    This model inputs and outputs to/from (s) space, while
    the intermediate representation is in z space
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, stacks, e_dim, heads, n_embeddings, embedding_dim, beta) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, stacks)
        in_dims = [h_dim, int(h_dim * 1.5), h_dim * 2]
        out_dims = [int(h_dim * 1.5), h_dim * 2, h_dim * 4]
        down_conv = []
        for _in_dim, out_dim in zip_longest(in_dims, out_dims):
            down_conv.append(nn.Conv2d(_in_dim, out_dim, 4, 2, 1))
            down_conv.append(nn.BatchNorm2d(out_dim))
            down_conv.append(nn.LeakyReLU(True))

        self.down_conv_stack = nn.Sequential(*down_conv)
        code_dim = 2048
        self.time_embedding = TimeEmbedding2D(10, e_dim)
        self.mlp_up = torch.nn.Parameter(_xav(torch.randn(1, e_dim + code_dim, code_dim * 2)), requires_grad=True)
        self.mlp_bias = torch.nn.Parameter(_xav(torch.randn(1, code_dim * 2)), requires_grad=True)
        self.mlp_down = torch.nn.Parameter(_xav(torch.randn(1, code_dim * 2, code_dim)), requires_grad=True)
        up_conv = []
        up_conv.append(nn.ConvTranspose2d(h_dim * 4, h_dim * 2, kernel_size=4, stride=2, padding=1))
        up_conv.append(nn.BatchNorm2d(h_dim * 2))
        up_conv.append(nn.LeakyReLU(True))
        up_conv.append(nn.ConvTranspose2d(h_dim * 2, int(h_dim * 1.5), 4, 2, 1))
        up_conv.append(nn.BatchNorm2d(int(h_dim * 1.5)))
        up_conv.append(nn.LeakyReLU(True))
        up_conv.append(nn.ConvTranspose2d(int(h_dim * 1.5), h_dim, 4, 2, 1))
        up_conv.append(nn.BatchNorm2d(h_dim))
        up_conv.append(nn.LeakyReLU(True))
        self.up_conv_stack = nn.Sequential(*up_conv)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, stacks, in_dim)
        self.in_dim = in_dim

    def forward(self, s_x, t_xy):
        """
        s_x -> z_x
        z_x, t_xy -> z_y (mlp residual)
        z_y -> s_y

        Returns:
            tuple: (s_y, z_x, z_y)
        """
        z = self.down_conv_stack(self.encoder(s_x))
        z_shape = z.shape
        lc = z.flatten(1)
        t_xy = self.time_embedding(t_xy.flatten(1))
        tlc = torch.cat([t_xy, lc], dim=-1)
        affine = fn.leaky_relu_((tlc @ self.mlp_up) + self.mlp_bias)
        lc = lc + (affine @ self.mlp_down)
        z = lc.reshape(*z_shape)
        z = self.up_conv_stack(z)
        z = self.pre_quantization_conv(z)
        embed_loss, z_q, perp, _, _ = self.vector_quantization(z)
        return embed_loss, self.decoder(z_q), perp, lc


class BigModel(nn.Module):
    def __init__(self, outer: VQVAE, inner: TSVAE):
        super().__init__()
        self.outer = outer
        self.inner = inner
        self.pass_thru = nn.Conv2d(inner.in_dim * 2, inner.in_dim, 3, 1, 1)
        self.adapter = ResidualStack(inner.in_dim, inner.in_dim, 128, 8)
        for p in self.adapter.parameters():
            torch.nn.init.zeros_(p)

    def forward(self, x, t_xy):
        """
        args:
            x: input image
            t_xy: time difference between x and y
        return:
            y_hat: reconstructed output image
            embed_loss_inner: VQ embedding loss for inner model
            embed_loss_outer: VQ embedding loss for outer model
            perplexity_inner: VQ codebook usage for inner model
            perplexity_outer: VQ codebook usage for outer model
            s_y: structural latent of y
        """
        s_x = self.outer.generate_latent(x)
        embed_loss_inner, s_y, perplexity_inner, _ = self.inner(s_x, t_xy)
        s_x = self.adapter(s_x)
        s = self.pass_thru(torch.cat([s_x, s_y], dim=1))
        embed_loss_outer, y_hat, perplexity_outer, _ = self.outer.generate_output_from_latent(s)
        return y_hat, embed_loss_inner, embed_loss_outer, perplexity_inner, perplexity_outer, s_y

    @torch.compile(mode='max-autotune', fullgraph=True)
    def no_pass_thru(self, x, t_xy):
        s_x = self.outer.generate_latent(x)
        embed_loss_inner, s_y, perplexity_inner, _ = self.inner(s_x, t_xy)
        embed_loss_outer, y_hat, perplexity_outer, _ = self.outer.generate_output_from_latent(s_y)
        return y_hat, embed_loss_inner, embed_loss_outer, perplexity_inner, perplexity_outer, s_y

    def outer_pass(self, x):
        """
        return:
            embedding_loss: VQ embedding loss
            x_hat: reconstructed output image
            perplexity: VQ codebook usage
            s_x: structural latent of x
        """
        return self.outer(x)

    def inner_pass(self, s_x, t_xy):
        """
        return:
            s_y: structural latent for y
            z_x: latent for x
            z_y: latent for y
        """
        return self.inner(s_x, t_xy)


real_label = 1
fake_label = 0


def to_params(tensors, init='none'):
    params = [nn.Parameter(t, requires_grad=True) for t in tensors]
    if init == 'normal':
        for p in params:
            torch.nn.init.normal_(p, mean=0.0, std=0.2)
    return nn.ParameterList(params)


# -------------- Functions


def save_model(model):
    artifact = wandb.Artifact(args.name, type='model')
    artifact.add_file(local_path=args.config_file, name='model_config', is_tmp=True)
    checkpoint_path = Path('./.checkpoints') / f'{args.name}'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f'{str(checkpoint_path.resolve())}/model_state_dict.pth'
    torch.save(model.state_dict(), checkpoint_path)
    artifact.add_file(local_path=checkpoint_path, name='model_state_dict', is_tmp=True)
    wandb.log_artifact(artifact)


def training_step(batch_idx, batch):
    global full_train_flag
    x, y, t = unpack(batch, device)

    if full_train_flag:
        y_hat, embed_i, embed_o, perp_i, perp_o, _ = model(x, t)
    else:
        y_hat, embed_i, embed_o, perp_i, perp_o, _ = model.no_pass_thru(x, t)

    recon_loss = ssim_loss(y_hat, y)
    loss = recon_loss + embed_o + embed_i

    optim.zero_grad()
    loss.backward()
    optim.step()

    if batch_idx % logging_rate == 0:
        psnr_x = psnr(y_hat, y)
        ssim_x = ssim(y_hat, y)
        msssim_x = ms_ssim(y_hat, y)

        wandb.log(
            {
                'train/recon_loss': recon_loss.item(),
                'train/perplexity': perp_o.item(),
                'train/perplexity_inner': perp_i.item(),
                'train/embed_loss_out': embed_o.item(),
                'train/embed_loss_in': embed_i.item(),
                'train/pred_psnr': psnr_x.item(),
                'train/pred_ssim': ssim_x.item(),
                'train/pred_ms-ssim': msssim_x.item(),
            }
        )

    if batch_idx % img_logging_rate == 0:
        caption = 'x, y_hat (predicted), y'
        mosaic = torch.cat([x[:4], y_hat[:4], y[:4]], dim=-1)
        wandb.log({'train/images': [wandb.Image(img, caption=caption) for img in mosaic]})


def validation_step(batch_idx, batch):
    global full_train_flag
    x, y, t = unpack(batch, device)
    with torch.no_grad():
        if full_train_flag:
            y_hat, embed_i, embed_o, perp_i, perp_o, _ = model(x, t)
        else:
            y_hat, embed_i, embed_o, perp_i, perp_o, _ = model.no_pass_thru(x, t)

        recon_loss = ssim_loss(y_hat, y)
        loss = recon_loss + embed_o + embed_i

        psnr_x = psnr(y_hat, y)
        ssim_x = ssim(y_hat, y)
        msssim_x = ms_ssim(y_hat, y)

    return recon_loss, perp_o, perp_i, psnr_x, ssim_x, msssim_x


def warmup_vqvae(batch_idx, batch):
    x, y, _ = unpack(batch, device)

    embed_loss_x, x_hat, perp_x, _ = model.outer_pass(x)
    embed_loss_y, y_hat, perp_y, _ = model.outer_pass(y)
    recon_loss_x, recon_loss_y = ssim_loss(x_hat, x), ssim_loss(y_hat, y)
    outer_loss = recon_loss_x + recon_loss_y + embed_loss_x + embed_loss_y

    loss = outer_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    if batch_idx % logging_rate == 0:
        psnr_x = psnr(y_hat, y)
        ssim_x = ssim(y_hat, y)
        msssim_x = ms_ssim(y_hat, y)
        mean_perp = (perp_x.item() + perp_y.item()) / 2
        mean_recon = (recon_loss_x.item() + recon_loss_y.item()) / 2
        mean_embed = (embed_loss_x.item() + embed_loss_y.item()) / 2
        wandb.log(
            {
                'train/outer_loss': outer_loss.item(),
                'train/recon_loss': mean_recon,
                'train/perplexity': mean_perp,
                'train/embed_loss': mean_embed,
                'train/psnr': psnr_x.item(),
                'train/ssim': ssim_x.item(),
                'train/ms-ssim': msssim_x.item(),
            }
        )

    if batch_idx % 2500 == 0:
        caption = 'x, x_hat, y_hat (no inner), y'
        mosaic = torch.cat([x[:4], x_hat[:4], y_hat[:4], y[:4]], dim=-1)
        wandb.log({'train/warmup_images': [wandb.Image(img, caption=caption) for img in mosaic]})


@torch.no_grad
def running_average_weights(model: nn.Module, path, beta):
    with torch.no_grad():
        state = torch.load(path, weights_only=False).state_dict()
        model_state = model.state_dict()
        for name in model_state.keys():
            model_state[name].data = (model_state[name].data * beta) + (state[name].data * (1 - beta))
        torch.save(model, path)
        model.load_state_dict(model_state)


# --------------- Script
if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='train the timescale diffusion model')
    parser.add_argument('config_file', help='Path to the configuration file')
    parser.add_argument('--name', help='run name.')
    args = parser.parse_args()

    # Load config
    config = toml.decoder.load(args.config_file)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    # Hyperparameters
    batch_size = config['data']['batch_size']
    learning_rate = config['hp']['lr'] if 'lr' in config['hp'] else 0.001
    warmup_steps = config['hp']['warmup'] if 'warmup_steps' in config['hp'] else 10000
    num_epochs = config['hp']['num_epochs'] if 'num_epochs' in config['hp'] else 5
    svd_alpha = config['hp']['svd_alpha'] if 'svd_alpha' in config['hp'] else 0.1
    epoch_size = config['hp']['epoch_size'] if 'epoch_size' in config['hp'] else 100000
    logging_rate = config['hp']['logging_rate'] if 'logging_rate' in config['hp'] else 50
    img_logging_rate = config['hp']['img_logging_rate'] if 'img_logging_rate' in config['hp'] else logging_rate**2
    val_interval = config['hp']['val_interval'] if 'val_interval' in config['hp'] else 1000
    full_train_flag = False

    # Model(s)
    _outer_model = VQVAE(**config['outer'])
    _inner_model = TSVAE(**config['inner'])
    _model = BigModel(_outer_model, _inner_model)
    summary(_model, device=device, depth=4, input_size=((batch_size, 3, 256, 256), (batch_size, 2, 10)))
    _model.to(memory_format=torch.channels_last)
    model = torch.compile(_model, **config['compile'])

    # optim
    optim = schedulefree.AdamWScheduleFree(_model.parameters(), lr=learning_rate, warmup_steps=warmup_steps)

    # loss
    ssim_loss = MixReconstructionLoss()

    # img quality metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    # dataset
    dataset = PairDataset(**config['data'])
    val_dataset = PairDataset(**config['val_data'])
    VAL_BATCHES = 50
    # wandb
    wandb.init(project='timescale-diffusion', name=args.name)
    save_model(_model)

    e = 0
    total_idx = 0
    while e < num_epochs:
        wandb.log({'epoch': e})
        try:
            for batch_idx, batch in tqdm(islice(enumerate(dataset), 50000), total=50000, desc='warmup'):
                warmup_vqvae(batch_idx, batch)

            for batch_idx, batch in tqdm(enumerate(dataset), total=epoch_size, mininterval=0.5):
                training_step(batch_idx, batch)
                if total_idx >= 250_000:
                    full_train_flag = True
                total_idx += 1
                if batch_idx >= epoch_size:
                    save_model(_model)
                    break
                if total_idx % val_interval == 0:
                    recon_loss = 0.0
                    perplexity = 0.0
                    perplexity_inner = 0.0
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_ms_ssim = 0.0

                    optim.eval()
                    model.eval()

                    for val_batch in tqdm(islice(val_dataset, VAL_BATCHES), total=VAL_BATCHES, desc='validation'):
                        loss, perp_o, perp_i, psnr_val, ssim_val, ms_ssim_val = validation_step(batch_idx, batch)
                        recon_loss += loss.item()
                        perplexity += perp_o.item()
                        perplexity_inner += perp_i.item()
                        avg_psnr += psnr_val.item()
                        avg_ssim += ssim_val.item()
                        avg_ms_ssim += ms_ssim_val.item()

                    wandb.log(
                        {
                            'val/recon_loss': recon_loss / VAL_BATCHES,
                            'val/perplexity': perplexity / VAL_BATCHES,
                            'val/perplexity_inner': perplexity_inner / VAL_BATCHES,
                            'val/pred_psnr': avg_psnr / VAL_BATCHES,
                            'val/pred_ssim': avg_ssim / VAL_BATCHES,
                            'val/pred_ms-ssim': avg_ms_ssim / VAL_BATCHES,
                        }
                    )
                    optim.train()
                    model.train()

        except Exception as ex:  # noqa: E722
            print(ex)
            save_model(_model)
            del dataset
            gc.collect()
            dataset = PairDataset(**config['data'])
            print('Dataloader crashed, restarting epoch.')
        else:
            e += 1
