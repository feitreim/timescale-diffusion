import torch
from torch import Tensor
from torch.cuda.random import device_count
import wandb
import os
from typing import List, Tuple
import toml

from vqvae.model import VQVAE
from diffusion_model import LTDM


def collate_timescales(
    batches: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    frames, timestamps = zip(*batches)  # unzip batches
    return torch.stack(frames), torch.Tensor(timestamps)


SECONDS = [
    1.0,  # Second
    60.0,  # Minute
    60 * 60.0,  # Hour
    60 * 60 * 24.0,  # Day
    60 * 60 * 24 * 7.0,  # Week
    60 * 60 * 24 * 30.42,  # Month
    60 * 60 * 24 * 365.0,  # Year
]

FRAMES = [
    30.0,  # Second
    60.0 * 30.0,  # Minute
    60 * 60.0 * 30.0,  # Hour
    60 * 60 * 24.0 * 30.0,  # Day
    60 * 60 * 24 * 7.0 * 30.0,  # Week
    60 * 60 * 24 * 30.42 * 30.0,  # Month
    60 * 60 * 24 * 365.0 * 30.0,  # Year
]

LEVEL = [
    30,
    60,
    60,
    24,
    7,
    31,
    10,
]


def convert_timestamp_to_periodic(t, fps=30, offset_seconds=0) -> Tensor:
    """
    Convert a timestamp give in frames, to a periodic representation.
    output will be a tensor of length 7, with each value being from
    [0, 1.0], for second, hour, day, week, month, year
    """
    offset = offset_seconds * fps
    timestamp = t + offset
    level = 0.1
    output_list = [
        int(((timestamp % FRAMES[i]) / FRAMES[i]) * LEVEL[i]) / LEVEL[i]
        for i in range(len(FRAMES))
    ]
    return torch.as_tensor(output_list)


def convert_timestamp_to_periodic_vec(t):
    pass


def unpack(batch, device) -> Tuple[Tensor, Tensor, Tensor]:
    x, y, t_x, t_y = batch
    x = x.to(device)
    y = y.to(device)
    t = torch.stack([t_x, t_y], dim=1).to(device)
    return x, y, t


def download_artifact(name: str) -> Tuple[str, str]:
    """
    downloads the artifact with the given name and then
    returns:
        - state_dict (str): path to the state dict.
        - arg_dict (str): path to the args dict.
    """
    api = wandb.Api()
    artifact = api.artifact(name)
    root = artifact.download()
    files = os.listdir(root)
    f1 = os.path.join(root, files[0])
    f2 = os.path.join(root, files[1])
    if os.path.getsize(f1) > os.path.getsize(f2):
        return f1, f2
    else:
        return f2, f1


def clear_gpu_mem_after(func):
    def wrapper(*args):
        func(*args)
        torch.cuda.empty_cache()

    return wrapper


def load_frozen_vqvae(name: str):
    state, args = download_artifact(name)
    args_dict = toml.load(args)
    vqvae = VQVAE(**args_dict["model_options"])
    vqvae.load_state_dict(torch.load(state, map_location="cpu"))
    vqvae.eval()
    return vqvae


def load_model_from_artifact(artifact):
    state_dict_path, arg_dict_path = download_artifact(artifact)
    config = toml.load(arg_dict_path)
    model = LTDM(config["unet_config"], config["vqvae_config"])
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model


def non_ar_video(model, frame, start, end, step):
    s_time = convert_timestamp_to_periodic(start)
    frame = frame.squeeze().reshape(1, 3, 256, 256)
    frames = []
    for i in range(start, end, step):
        c_time = convert_timestamp_to_periodic(i)
        t = torch.stack([s_time, c_time]).unsqueeze(0)

        frames.append(model.diffusion_step(frame, t))
    return frames
