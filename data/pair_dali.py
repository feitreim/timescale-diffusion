import torch
import os
import numpy as np
import pytorch_lightning as pl
from typing import Tuple, Dict
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as pydali
from torch.utils.data import DataLoader
from einops import rearrange
import random
import dill

from utils import convert_timestamp_to_periodic
from data.frame_random_index import FrameRandomIndexModule


@pipeline_def(
    py_start_method="spawn",
)
def img_pipe(
    ext_source,
    fill,
    name,
    prefetch,
    shuffle,
    shard_id=0,
    num_devices=1,
):
    j2ks, t_x, t_y = fn.external_source(
        source=ext_source, batch=True, parallel=True, batch_info=True, num_outputs=3
    )

    images = fn.decoders.image(
        j2ks,
        preallocate_height_hint=256,
        preallocate_width_hint=256,
        use_fast_idct=True,
        device="mixed",
    )

    reorder = fn.transpose(images, perm=[2, 0, 1])
    scale = types.Constant(255)
    normalized = reorder / scale
    return normalized, t_x, t_y


class PairDataset(torch.utils.data.IterableDataset):
    """
    Iterator Style pytorch dataset that uses a NVIDIA DALI pipeline to load video data
    quickly with gpu acceleration. Also provides multi-frame context.

    Args:
        - video_file_paths: (List[str]): A list of all the paths to the data.
        - sequence_length: int: number of total frames for multi-frame context.
        - batch_size: int: batch size, number of multi-frame sets.
        - num_threads: int: number of cpu(?) threads for dataloading.
        - output_size: list[int, int]: Pipeline output image size.
        - buffer_size: int: size of the preloaded buffer
        - mf_distance: int: distance between sequence frames.
        - device_id: int: device id of the cuda device to run the dataloader on.
        - num_devices: int: number of total devices
        - shard_id: int: id of the shard.
    """

    def __init__(
        self,
        dir,
        periodic,
        batch_size,
        num_threads,
        fill,
        prefetch,
        shuffle,
        shard_id=0,
        num_devices=1,
    ):
        super().__init__()
        self.name = f"Reader{shard_id}"

        files = os.listdir(dir)
        random.shuffle(files)

        labels = [f.split("_") for f in files]
        x_labels = [convert_timestamp_to_periodic(int(l[0]), fps=30) for l in labels]
        y_labels = [convert_timestamp_to_periodic(int(l[1]), fps=30) for l in labels]
        n = len(files)

        globals().update(
            {
                "n": n,
                "dir": dir,
                "batch_size": batch_size,
                "files": files,
                "x_labels": x_labels,
                "y_labels": y_labels,
            }
        )

        def external_source(info):
            idx = info.iteration
            if idx + batch_size >= n:
                raise StopIteration
            batch = []
            t_x = []
            t_y = []
            for i in range(batch_size):
                f = open(dir + files[idx + i], "rb")
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
                t_x.append(np.array([x_labels[idx + i]], dtype=np.float32))
                t_y.append(np.array([y_labels[idx + i]], dtype=np.float32))
            return (batch, t_x, t_y)

        pipe = img_pipe(
            ext_source=external_source,
            fill=fill,
            prefetch=prefetch,
            shuffle=shuffle,
            name=self.name,
            shard_id=shard_id,
            num_devices=num_devices,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=shard_id,
        )

        self.pipeline = pydali.DALIGenericIterator(
            pipe, output_map=["frames", "t_x", "t_y"]
        )
        self.periodic = periodic

    def __iter__(self):
        for data in enumerate(self.pipeline):
            frames, t_x, t_y = (
                data[1][0]["frames"],
                data[1][0]["t_x"],
                data[1][0]["t_y"],
            )

            frames = frames.squeeze()
            x = frames[:, :, :, :256]
            y = frames[:, :, :, 256:]

            yield (x, y, t_x.squeeze(), t_y.squeeze())

    def __len__(self):
        return self.pipeline.size


class PairDatamodule(pl.LightningDataModule):
    """
    Sequenced Multiframe NVIDIA Dali powered dataloader with normalized
    timestamps accross the whole dataset.

    Args:
        - filelist: list of files w labels
        - periodic: periodic timestamp embeddings yes or no
        - batch_size: batch size
        - fill: how many imgs to fill in queue
        - prefetch: prefetch queue depth
        - num_devices: int: number of total devices
        - shard_id: int: id of the shard.
    """

    def __init__(
        self,
        filelist: str,
        periodic: bool,
        batch_size: int,
        num_threads: int,
        fill: int,
        prefetch: int,
        num_devices: int,
        val_dataloader: Dict,
    ):
        super().__init__()
        self.filelist = filelist
        self.periodic = periodic
        self.batch_size = batch_size
        self.fill = fill
        self.prefetch = prefetch
        self.num_threads = num_threads
        self.num_devices = num_devices
        self.val_dict = val_dataloader

    def setup(self, stage):
        super().setup(stage=stage)
        self.local_rank = int(os.environ["LOCAL_RANK"])

        self.name = f"Reader{self.local_rank}"
        pipe = img_pipe(
            filelist=self.filelist,
            fill=self.fill,
            prefetch=self.prefetch,
            shuffle=True,
            name=self.name,
            shard_id=self.local_rank,
            num_devices=self.num_devices,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.local_rank,
        )

        class LightningWrapper(pydali.DALIGenericIterator):
            def __init__(self, periodic, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)
                self.periodic = periodic

            def __next__(self):
                out = super().__next__()
                labels = out[0]["labels"]
                frames = out[0]["frames"]

                if self.periodic:
                    label_list = [
                        convert_timestamp_to_periodic(label) for label in labels
                    ]
                    labels = torch.stack(label_list)

                return frames, labels

        self.loader = LightningWrapper(
            self.periodic, pipe, output_map=["frames", "labels"], reader_name=self.name
        )

    def train_dataloader(self):
        return self.loader

    def test_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return FrameRandomIndexModule(**self.val_dict).val_dataloader()
