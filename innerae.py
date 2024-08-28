import argparse
import random
from pathlib import Path

import toml
import gc
import torch
import torch.nn as nn
import torch.nn.functional as fn
import wandb
import schedulefree
from torchinfo import summary
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from data.pair_dali import PairDataset
from losses.recon import MixReconstructionLoss
from utils import unpack, convert_timestamp_to_periodic_vec
from vqvae.model import VQVAE
from vqvae.blocks.encoder import Encoder
from vqvae.blocks.decoder import Decoder
from tspm.attention import AttnBlock
from tspm.time import TimeEmbedding2D


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

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, stacks, e_dim, heads) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim, n_res_layers, res_h_dim, stacks)
        self.time_embedding = TimeEmbedding2D(10, e_dim)
        self.attention = [AttnBlock(e_dim) for _ in range(heads)]
        self.decoder = Decoder(h_dim, h_dim, n_res_layers, res_h_dim, stacks, in_dim)

    def forward(self, s_x, t_xy):
        """
        s_x -> z_x
        z_x, t_xy -> z_y ( cross attn )
        z_y -> s_y

        out: ( s_y, z_x, z_y )
        """
        z = self.encoder(s_x)
        z_x = z.clone()
        t_xy = self.time_embedding(t_xy.flatten(1))
        for attn in self.attention:
            z = attn(z, t_xy)
        return self.decoder(z), z_x, z


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
    x, y, t = unpack(batch, device)

    s_x = outer_model.generate_latent(x)
    s_y = outer_model.generate_latent(y)
    embed_loss_x, x_hat, perp_x, _ = outer_model.generate_output_from_latent(s_x)
    embed_loss_y, y_hat, perp_y, _ = outer_model.generate_output_from_latent(s_y)
    recon_loss_x, recon_loss_y = ssim_loss(x_hat, x), ssim_loss(y_hat, y)
    outer_loss = recon_loss_x + recon_loss_y + embed_loss_x + embed_loss_y

    outer_optim.zero_grad()
    outer_loss.backward()
    outer_optim.step()

    s_y = s_y.detach()
    s_y_hat, _, _ = inner_model(s_x, t)
    inner_loss = torch.pow(s_y - s_y_hat, 2.0).mean()

    inner_optim.zero_grad()
    inner_loss.backward()
    inner_optim.step()

    if batch_idx % logging_rate == 0:
        psnr_x = psnr(y_hat, y)
        ssim_x = ssim(y_hat, y)
        msssim_x = ms_ssim(y_hat, y)
        mean_perp = (perp_x.item() + perp_y.item()) / 2
        mean_recon = (recon_loss_x.item() + recon_loss_y.item()) / 2
        mean_embed = (embed_loss_x.item() + embed_loss_y.item()) / 2

        with torch.no_grad():
            _, recon, _, _ = outer_model.generate_output_from_latent(s_y_hat)
        psnr_i = psnr(recon, y)
        ssim_i = ssim(recon, y)
        msssim_i = ms_ssim(recon, y)

        wandb.log(
            {
                'train/outer_loss': outer_loss.item(),
                'train/recon_loss': mean_recon,
                'train/perplexity': mean_perp,
                'train/embed_loss': mean_embed,
                'train/psnr': psnr_x.item(),
                'train/ssim': ssim_x.item(),
                'train/ms-ssim': msssim_x.item(),
                'train/psnr_pred': psnr_i.item(),
                'train/ssim_pred': ssim_i.item(),
                'train/msssim_pred': msssim_i.item(),
            }
        )

    if batch_idx % img_logging_rate == 0:
        with torch.no_grad():
            _, recon, _, _ = outer_model.generate_output_from_latent(s_y_hat)
        caption = 'x, x_hat, y_hat (predicted), y_hat (no inner), y'
        mosaic = torch.cat([x[:4], x_hat[:4], recon[:4], y_hat[:4], y[:4]], dim=-1)
        wandb.log({'train/images': [wandb.Image(img, caption=caption) for img in mosaic]})


@torch.no_grad
def validation_step(
    batch_idx,
    batch,
    psnr,
):
    x, y, t = unpack(batch, device)

    (embed_loss_x, embed_loss_y, x_hat, y_hat, perp_x, perp_y) = model(x, t)

    loss = ssim_loss(x_hat, y)
    psnr_x = psnr(x_hat, x)
    psnr_y = psnr(y_hat, y)

    if batch_idx % 10 == 0:
        wandb.log({'val/loss': loss.item()})


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

    # Model(s)
    outer_model = VQVAE(**config['vqvae'])
    summary(
        outer_model,
        depth=4,
        input_size=(batch_size, 3, 256, 256),
    )
    outer_model = outer_model.to(device)

    inner_model = TSVAE(**config['tsvae'])
    summary(inner_model, depth=4, input_size=((batch_size, 64, 64, 64), (batch_size, 2, 10)))
    inner_model = inner_model.to(device)
    # optim
    outer_optim = schedulefree.AdamWScheduleFree(outer_model.parameters(), lr=learning_rate, warmup_steps=warmup_steps)
    inner_optim = schedulefree.AdamWScheduleFree(inner_model.parameters(), lr=learning_rate, warmup_steps=warmup_steps)
    # ema of weights
    ema_id = random.randint(0, 2000000)
    ema_path = f'./.running_avgs/{ema_id}/'
    ema_beta = config['ema']['beta']
    ema_enabled = config['ema']['enabled']
    ema_interval = config['ema']['interval']
    ema_start = config['ema']['start']
    Path(ema_path).mkdir(parents=True, exist_ok=True)
    ema_path = Path(ema_path) / 'lastweight.ckpt'
    torch.save(inner_model, ema_path)

    # loss
    ssim_loss = MixReconstructionLoss()

    # img quality metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    # dataset
    dataset = PairDataset(**config['data'])
    # val_dataset = FrameDataset(**config['val_data'])
    # wandb
    wandb.init(project='timescale-diffusion', name=args.name)
    save_model(inner_model)
    e = 0
    while e < num_epochs:
        wandb.log({'epoch': e})
        try:
            for batch_idx, batch in tqdm(enumerate(dataset)):
                training_step(batch_idx, batch)
                if batch_idx % ema_interval == 0 and batch_idx > ema_start and ema_enabled:
                    running_average_weights(inner_model, ema_path, ema_beta)
                elif batch_idx % ema_interval == 0 and ema_enabled:
                    torch.save(inner_model, ema_path)
                if batch_idx >= epoch_size:
                    save_model(inner_model)
                    break
        except Exception as ex:  # noqa: E722
            print(ex)
            save_model(inner_model)
            del dataset
            gc.collect()
            dataset = PairDataset(**config['data'])
            print('Dataloader crashed, restarting epoch.')
        else:
            e += 1
