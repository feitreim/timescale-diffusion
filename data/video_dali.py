import torch
import os
import pytorch_lightning as pl

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as pydali
from torch.utils.data import DataLoader
from utils import convert_timestamp_to_periodic

"""
Dali pipeline, allows gpu-accelerated dataloading
Args:
    - filenames: list[str]: list of the files for the dataloader.
    - sequence_length: int: multi-frame size.
    - output_size: list[int]: output dimensions for frames.
    - buffer_size: int: size of the dataloading buffer.
    - mf_distance: int: distance between multiframe-frames
"""


@pipeline_def
def video_pipeline(
    filenames,
    sequence_length,
    output_size,
    buffer_size,
    mf_distance,
    shard_id,
    num_shards,
    name,
    shuffle,
):
    frames, labels, frame_number = fn.readers.video_resize(
        device='gpu',
        additional_decode_surfaces=4,
        filenames=filenames,
        sequence_length=sequence_length,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=shuffle,
        initial_fill=buffer_size,
        enable_frame_num=True,
        stride=mf_distance,
        size=output_size,
        mode='stretch',
        minibatch_size=32,
        read_ahead=True,
        prefetch_queue_depth=2,
        name=name,
        labels=[],
    )

    # Image Transforms
    reordered = fn.transpose(frames, perm=[0, 3, 1, 2])  # SCHW
    scale = types.Constant(255)
    normalized = reordered / scale

    return normalized, labels, frame_number


class DALIDataset(torch.utils.data.IterableDataset):
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
        video_file_paths,
        sequence_length,
        batch_size,
        num_threads,
        output_size,
        buffer_size,
        mf_distance,
        num_devices=1,
        device_id=0,
        shard_id=0,
        shuffle=True,
    ):
        super().__init__()
        self.name = f'Reader{device_id}'
        pipe = video_pipeline(
            filenames=video_file_paths,
            sequence_length=sequence_length,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            output_size=output_size,
            buffer_size=buffer_size,
            mf_distance=mf_distance,
            shard_id=shard_id,
            num_shards=num_devices,
            name=self.name,
            shuffle=shuffle,
            prefetch_queue_depth=4,
            set_affinity=True,
        )
        self.pipeline = pydali.DALIGenericIterator(pipe, output_map=['frames', 'labels', 'frame_number'], reader_name=self.name)
        self.video_timestamps = []
        self.compute_frame_timestamps(video_file_paths)
        self.offsets = [i * mf_distance for i in range(sequence_length)]
        self.sq_len = sequence_length

    def __iter__(self):
        for data in enumerate(self.pipeline):
            frames, labels, frame_number = (
                data[1][0]['frames'],
                data[1][0]['labels'],
                data[1][0]['frame_number'],
            )

            # frames = rearrange(frames, "B S C H W -> (B S) C H W")
            timestamps = torch.empty((frames.shape[0], frames.shape[1], 7), dtype=torch.float32)
            for idx, l in enumerate(labels):
                label = l[0]
                base_timestamp = self.video_timestamps[label]
                timestamp_with_frame = base_timestamp + frame_number[idx]
                for i, off in enumerate(self.offsets):
                    timestamp_w_offset = timestamp_with_frame + off
                    timestamps[idx, i] = convert_timestamp_to_periodic(timestamp_w_offset, fps=30).squeeze()

            yield frames, timestamps

    def __len__(self):
        return self.pipeline.size

    def compute_frame_timestamps(self, video_file_paths):
        """
        Compute proper timestamps for each video from oldest->newest.
        Pre:
            - Videos should already be in order.
            - Videos should contain timestamp information in YYMMDD_HHMMSS
        """
        start = []
        self.video_timestamps = []

        for path in video_file_paths:
            # Isolate date/time strings
            filename = path.split('/')[-1]
            filename = filename.casefold().strip("adcdefghijklmnopqrstuvwxyz,.;'[]{}:<>?/")
            yearmonthday = filename.split('_')[0]
            hourminsec = filename.split('_')[-1]

            # Create substring of all info
            date_info = []

            date_info.append(int(yearmonthday[0:2]))
            date_info.append(int(yearmonthday[2:4]))
            date_info.append(int(yearmonthday[4:6]))
            date_info.append(int(hourminsec[0:2]))
            date_info.append(int(hourminsec[2:4]))
            date_info.append(int(hourminsec[4:6]))

            # Set a start date
            if len(start) < 1:
                for x in date_info:
                    start.append(x)

            # Calculate offset in frames
            years = date_info[0] - start[0]
            months = (date_info[1] - start[1]) + (years * 12)
            days = (date_info[2] - start[2]) + (months * 30)
            hours = (date_info[3] - start[3]) + (days * 24)
            minutes = (date_info[4] - start[4]) + (hours * 60)
            seconds = (date_info[5] - start[5]) + (minutes * 60)
            frames = seconds * 30

            self.video_timestamps.append(frames)
        print(self.video_timestamps)


class LightningDaliLoader(pl.LightningDataModule):
    """
    Sequenced Multiframe NVIDIA Dali powered dataloader with normalized
    timestamps accross the whole dataset.

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
        video_file_paths,
        sequence_length,
        batch_size,
        num_threads,
        output_size,
        buffer_size,
        mf_distance,
        num_devices,
    ):
        super().__init__()

        self.video_file_paths = video_file_paths
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.output_size = output_size
        self.buffer_size = buffer_size
        self.mf_distance = mf_distance
        self.num_devices = num_devices
        self.prepare_data_per_node = False

    def setup(self, stage):
        super().setup(stage=stage)
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.dataset = DALIDataset(
            self.video_file_paths,
            self.sequence_length,
            self.batch_size,
            self.num_threads,
            self.output_size,
            self.buffer_size,
            self.mf_distance,
            self.num_devices,
            self.local_rank,
            self.local_rank,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset)

    def test_dataloader(self):
        return DataLoader(self.dataset)

    def val_dataloader(self):
        return DataLoader(self.dataset)
