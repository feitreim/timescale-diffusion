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
from vqvae.blocks.encoder import Encoder
from vqvae.blocks.quantizer import VectorQuantizer
from vqvae.blocks.structdec import StructuralDecoder


# -------------- Model
class SVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, stacks, n_embeddings, embedding_dim, beta) -> None:
        super().__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, 3)
        self.pre_time = Encoder(h_dim, h_dim, 0, res_h_dim, 1)
        in_feat = h_dim * 16 * 16
        self.latent_dense = nn.Linear(in_feat, 10)
        self.e_dim = embedding_dim
        self.pre_quantization_conv = nn.Parameter(torch.randn((embedding_dim, h_dim, 1, 1)), requires_grad=True)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = StructuralDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim, 3)

    def forward(self, x):
        z = self.encoder(x)
        z_t = self.pre_time(z)
        t_hat = self.to_latent(z_t)
        t_hat = t_hat.unsqueeze(1).expand(-1, 2, -1)
        z_pq = fn.conv2d(z, self.pre_quantization_conv)
        embedding_loss, z_q, perplexity, _, _ = self.vq(z_pq)
        y_hat = self.decoder(z_q, t_hat)
        return embedding_loss, y_hat, perplexity, z_q

    def to_latent(self, z):
        return fn.tanh(self.latent_dense(z.flatten(1)))

    @torch.compile(mode='max-autotune', fullgraph=True)
    def generate_latent_and_timecode(self, x):
        z = self.encoder(x)
        z_t = self.pre_time(z)
        t_hat = self.to_latent(z_t)
        return z, t_hat

    @torch.compile(mode='max-autotune', fullgraph=True)
    def predict(self, z, t):
        z_pq = fn.conv2d(z, self.pre_quantization_conv)
        embedding_loss, z_q, perplexity, _, _ = self.vq(z_pq)
        y_hat = self.decoder(z_q, t)
        return embedding_loss, y_hat, perplexity, z_q

    def generate_sequence(self, start, end, step):
        images = []
        for t in tqdm(range(start, end, step)):
            time = convert_timestamp_to_periodic_vec(torch.as_tensor([t]))
            _, img, _, _ = self.generate_image_from_timecode(time)
            images.append(img)
        return torch.cat(images, dim=0)


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
    t_x = t[:, 0]

    z, t_hat = model.generate_latent_and_timecode(x)
    embed_loss_x, y_hat, perp_x, z = model.predict(z, t)

    time_loss = torch.pow((t_x - t_hat), 2.0).mean()
    orig_loss = ssim_loss(y_hat, y)
    loss = orig_loss + embed_loss_x + time_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    if batch_idx % logging_rate == 0:
        psnr_x = psnr(y_hat, y)
        ssim_x = ssim(y_hat, y)
        msssim_x = ms_ssim(y_hat, y)

        wandb.log(
            {
                'train/loss': loss.item(),
                'train/recon_loss': orig_loss.item(),
                'train/perplexity': perp_x.item(),
                'train/embed_loss': embed_loss_x.item(),
                'train/time_loss': time_loss.item(),
                'train/psnr': psnr_x.item(),
                'train/ssim': ssim_x.item(),
                'train/ms-ssim': msssim_x.item(),
            }
        )

    if batch_idx % img_logging_rate == 0:
        caption = 'left: input, mid left: recon orig, mid right: recon target, right: target'
        mosaic = torch.cat([x[:4], y_hat[:4], y[:4]], dim=-1)
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
    model = SVAE(**config['vqvae'])
    summary(
        model,
        depth=4,
        input_size=(batch_size, 3, 256, 256),
    )
    model = model.to(device)

    # optim
    optim = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate, warmup_steps=warmup_steps)

    # ema of weights
    ema_id = random.randint(0, 2000000)
    ema_path = f'./.running_avgs/{ema_id}/'
    ema_beta = config['ema']['beta']
    ema_enabled = config['ema']['enabled']
    ema_interval = config['ema']['interval']
    ema_start = config['ema']['start']
    Path(ema_path).mkdir(parents=True, exist_ok=True)
    ema_path = Path(ema_path) / 'lastweight.ckpt'
    torch.save(model, ema_path)

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
    save_model(model)
    e = 0
    while e < num_epochs:
        wandb.log({'epoch': e})
        try:
            for batch_idx, batch in tqdm(enumerate(dataset)):
                training_step(batch_idx, batch)
                if batch_idx % ema_interval == 0 and batch_idx > ema_start and ema_enabled:
                    running_average_weights(model, ema_path, ema_beta)
                elif batch_idx % ema_interval == 0 and ema_enabled:
                    torch.save(model, ema_path)
                if batch_idx >= epoch_size:
                    save_model(model)
                    break
        except Exception as ex:  # noqa: E722
            print(ex)
            save_model(model)
            del dataset
            gc.collect()
            dataset = PairDataset(**config['data'])
            print('Dataloader crashed, restarting epoch.')
        else:
            e += 1
