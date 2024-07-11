import torch
from torch import Tensor
from torch.cuda.random import device_count
import wandb
import os
from typing import List, Tuple


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


def convert_timestamp_to_periodic(t, fps=30, offset_seconds=0) -> Tensor:
    """
    Convert a timestamp give in frames, to a periodic representation.
    output will be a tensor of length 7, with each value being from
    [0, 1.0], for second, hour, day, week, month, year
    """
    offset = offset_seconds * fps
    timestamp = t + offset
    output_list = [(timestamp % f) / f for f in FRAMES]
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