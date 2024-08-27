from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.utils.data
import pytorch_lightning as pl
import torchvision.transforms.v2 as v2
import torchvision.io as io
import os

from utils import convert_timestamp_to_periodic


class FrameDataset(torch.utils.data.IterableDataset):
    """
    Frame dataset for video pyramids.
    """

    def __init__(
        self,
        data_dir: str,
        input_size: List[int] = [256, 256],
        fps: int = 30,
        offset_seconds: int = 0,
        to_diff: bool = True,
        ordered: bool = False,
        periodic_timestamp=True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.input_size = input_size.copy()
        self.input_size[0] *= 5
        self.fps = fps
        self.offset_seconds = offset_seconds
        self.periodic_timestamp = periodic_timestamp

        if to_diff:
            self.transforms = v2.Compose(
                [
                    v2.ToDtype(torch.float32),
                    v2.Resize(self.input_size),
                    v2.Lambda(lambda x: ((x / 255.0) * 2) - 1),  # back to diff land
                ]
            )
        else:
            self.transforms = v2.Compose(
                [
                    v2.ToDtype(torch.float32),
                    v2.Resize(self.input_size),
                    v2.Lambda(lambda x: x / 255.0),
                ]
            )
        self.to_diff = to_diff

        self.total_frames = len(os.listdir(data_dir))
        self.frame_fnames = os.listdir(data_dir)

        if not ordered:
            self.frame_order = torch.randperm(self.total_frames)
        elif ordered:
            frames_as_nums = torch.tensor([int(f) for f in self.frame_fnames])
            _, indices = torch.sort(frames_as_nums)
            self.frame_order = indices

        self.current_index = 0

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        while self.current_index < self.total_frames:
            frame, timestamp = self.get_next_frame()
            frame = self.transforms(frame)
            if self.input_size[0] != self.input_size[1]:
                frame = torch.stack(torch.split(frame, self.input_size[-1], dim=-2), dim=1)
            pyr = frame.clone()  # [ B T C H W ]
            frame = frame.sum(dim=1)  # [ B C H W ]
            yield frame, pyr, timestamp

    def get_next_frame(self) -> Tuple[Tensor, Tensor]:
        idx = self.frame_order[self.current_index]
        img_path = f'{self.data_dir}{self.frame_fnames[idx]}'
        frame = io.read_image(img_path)
        self.current_index += 1
        if self.periodic_timestamp:
            timestamp = convert_timestamp_to_periodic(int(self.frame_fnames[idx]), self.fps, self.offset_seconds)
        else:
            timestamp = torch.as_tensor(int(self.frame_fnames[idx]))
        return frame, timestamp


class FrameRandomIndexModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        input_size: List[int] = [256, 256],
        fps: int = 30,
        offset_seconds: int = 0,
        batch_size: int = 24,
        num_workers: int = 8,
        val_data_dir: Union[str, None] = None,
        periodic_timestamp: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.output_shape = input_size
        self.fps = fps
        self.offset_seconds = offset_seconds
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.periodic_timestamp = periodic_timestamp

        if val_data_dir == None:
            self.val_data_dir = data_dir
        else:
            self.val_data_dir = val_data_dir

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            FrameDataset(
                self.data_dir,
                self.output_shape,
                self.fps,
                self.offset_seconds,
                periodic_timestamp=self.periodic_timestamp,
            ),
            collate_fn=collate_pyrs,
            batch_size=self.batch_size,
            prefetch_factor=4,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        # Same as training dataloader, except no data augmentations and batch size of 1
        return torch.utils.data.DataLoader(
            FrameDataset(
                self.val_data_dir,
                self.output_shape,
                self.fps,
                self.offset_seconds,
                ordered=True,
                periodic_timestamp=self.periodic_timestamp,
            ),
            collate_fn=collate_pyrs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        # TODO currently the same as the validation dataloader
        return torch.utils.data.DataLoader(
            FrameDataset(
                self.data_dir,
                self.output_shape,
                self.fps,
                self.offset_seconds,
                periodic_timestamp=self.periodic_timestamp,
            ),
            collate_fn=collate_pyrs,
            batch_size=self.batch_size,
            num_workers=1,
            drop_last=True,
        )


# ------------------------------- collate fn ----------------------------------
def collate_pyrs(batches: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    frames, pyrs, timestamps = zip(*batches)
    pyrs_batch = torch.stack(pyrs).permute(0, 2, 1, 3, 4)
    timestamps_batch = torch.stack(timestamps)
    frames_batch = torch.stack(frames)
    return frames_batch, pyrs_batch, timestamps_batch
