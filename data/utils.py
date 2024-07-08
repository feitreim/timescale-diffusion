import torch
from torch import Tensor

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

def unpack(batch) -> Tuple[Tensor, Tensor, Tensor]:
    x, t = batch
    x = x.squeeze()
    t = t.squeeze()
    y = x[:, 1]
    x = x[:, 0]
    t = t[:, 0]
    return x, y, t
