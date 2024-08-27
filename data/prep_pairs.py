import torchvision
import torch
import queue
import argparse
import os
import toml
import random
import multiprocessing as mp
import cv2

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

VLEN = 432_000  # Average lenght of a video


@dataclass
class VideoPair:
    lhs: int
    rhs: int
    pairs: List[Tuple[int, int]]
    lhs_offset: int
    rhs_offset: int
    lhs_path: str
    rhs_path: str


def get_frame_count(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print('Error opening video file')
        return -1

    # Get the total number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video object
    video.release()
    del video
    return frame_count


def process_pair(pair: VideoPair, root):
    print('processing pair.')
    l_video = torchvision.io.VideoReader(pair.lhs_path, 'video')
    l_video.set_current_stream('video:0')
    print('l_video open')
    r_video = torchvision.io.VideoReader(pair.rhs_path, 'video')
    print('r_video open')
    r_video.set_current_stream('video:0')

    l_len = get_frame_count(pair.lhs_path)
    fps = 30
    r_len = get_frame_count(pair.rhs_path)
    t_len = int(l_len + r_len)

    sub_dir = Path(root) / str(pair.lhs_offset)
    os.makedirs(sub_dir, exist_ok=True)

    for l_idx, r_idx in tqdm(pair.pairs):
        if l_idx < l_len and r_idx < r_len:
            l_video.seek(l_idx / fps)
            r_video.seek(r_idx / fps)
            l_frame, r_frame = next(l_video)['data'], next(r_video)['data']
            combined = torch.cat([l_frame, r_frame], dim=-1)
            fname = sub_dir / f'{pair.lhs_offset + l_idx}_{pair.rhs_offset + r_idx}.jpg'
            torchvision.io.write_jpeg(combined, fname, quality=100)

    while i < l_len and i + max_distance < t_len:
        distance = random.randint(0, max_distance)
        if i + distance >= l_len:
            l_video.seek(i / fps)
            l_frame = next(l_video)['data']
            idx = (i + distance - l_len) / fps
            idx = idx if idx > 0 else 1
            r_video.seek(idx)
            r_frame = next(r_video)['data']
        else:
            l_video.seek(i / fps)
            l_frame = next(l_video)['data']
            l_video.seek((i + distance) / fps)
            r_frame = next(l_video)['data']
        combined = torch.cat([l_frame, r_frame], dim=-1)
        fname = sub_dir / f'{offset+i}_{offset+i+distance}.jpg'
        torchvision.io.write_jpeg(combined, fname, quality=100)
        i += step + random.randint(-step // 2, step // 2)
        total += 1
        print(f'saved file {fname}. this proc saved {total}')

    return total


def likely_exists(index, offsets, start) -> Tuple[int, int]:
    for i in range(start - 1, len(offsets)):
        diff = index - offsets[i]
        if diff >= 0 and diff <= VLEN:
            return (diff, i)
    return (-1, -1)


def video_loop(current, pairs, offsets, distances, step):
    idx = 0
    rhs_tots = [0] * len(offsets)
    while idx <= VLEN:
        d = distances[random.randrange(0, len(distances))]

        if (rhs := likely_exists(idx + d, offsets, current))[0] >= 0:
            local_frame, video_idx = rhs
            pairs[current, video_idx, rhs_tots[video_idx]] = torch.as_tensor([idx, local_frame])
            rhs_tots[video_idx] += 1

        idx += step + random.randint(-step // 2, step // 2)


def compute_pair_from_tensor(pair_tensor, i, j):
    pairs = []
    for pair in pair_tensor[i, j]:
        pairs.append([pair[0], pair[1]])
    return pairs


def compute_video_pairs(videos, offsets, distances, step):
    length = len(videos)
    pairs = torch.ones((length, length, 2 * (VLEN // step), 2))
    pairs *= -1
    for v in range(length):
        video_loop(v, pairs, offsets, distances, step)

    v_pairs_list = []
    for lhs in range(length):
        for rhs in range(length):
            if pairs[lhs, rhs, 0, 0] != -1:
                v_pairs_list.append(
                    VideoPair(
                        lhs=lhs,
                        rhs=rhs,
                        pairs=compute_pair_from_tensor(pairs, lhs, rhs),
                        lhs_offset=offsets[lhs],
                        rhs_offset=offsets[rhs],
                        lhs_path=videos[lhs],
                        rhs_path=videos[rhs],
                    )
                )
    return v_pairs_list


def print_pairs(video_pairs):
    for pair in video_pairs:
        if pair.pairs[0][0] != -1:
            print(f'{pair.lhs}, {pair.rhs}, len: {len(pair.pairs)}')


def compute_frame_timestamps(first_vid, video_file_paths):
    """
    Compute proper timestamps for each video from oldest->newest.
    Pre:
        - Videos should already be in order.
        - Videos should contain timestamp information in YYMMDD_HHMMSS
    """

    def compute_date_info(path):
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
        return date_info

    start = compute_date_info(first_vid)
    video_timestamps = []

    for path in video_file_paths:
        date_info = compute_date_info(path)

        # Calculate offset in frames
        years = date_info[0] - start[0]
        months = (date_info[1] - start[1]) + (years * 12)
        days = (date_info[2] - start[2]) + (months * 30)
        hours = (date_info[3] - start[3]) + (days * 24)
        minutes = (date_info[4] - start[4]) + (hours * 60)
        seconds = (date_info[5] - start[5]) + (minutes * 60)
        frames = seconds * 30

        video_timestamps.append(frames)
    return video_timestamps


def gen_args(lhs, rhs, distance, step, offsets, output_path):
    for i in range(len(lhs)):
        args = (lhs[i], rhs[i], distance, step, offsets[i], output_path)
        yield args


def main():
    parser = argparse.ArgumentParser(description='train the timescale diffusion model')
    parser.add_argument('config_file', help='Path to the configuration file')
    parser.add_argument('output_path', help='run name.')
    parser.add_argument('--num_shards', help='total number of shards', type=int)
    parser.add_argument('--shard', help='which shard this is.', type=int)
    parser.add_argument('--step', help='step size', type=int, default=100)
    args = parser.parse_args()
    config = toml.decoder.load(args.config_file)

    vids = config['videos']
    offsets = compute_frame_timestamps(vids[0], vids)

    distances = config['distances']

    pairs = compute_video_pairs(vids, offsets, distances, args.step)

    # sharding
    shard_size = len(pairs) // args.num_shards
    start = shard_size * args.shard
    end = len(pairs)
    if not args.shard == args.num_shards - 1:
        end = shard_size * (args.shard + 1)

    pairs = pairs[start:end]
    print_pairs(pairs)

    print(f'shard size: {shard_size}, s: {start}, e: {end}')

    with mp.Pool(4) as pool:
        result_queue = mp.Manager().Queue()
        args_list = [
            (
                lhs_vids[i],
                rhs_vids[i],
                args.distance,
                args.step,
                offsets[i],
                args.output_path,
            )
            for i in range(len(lhs_vids))
        ]
        for result in pool.starmap(process_pair, args_list):
            result_queue.put(result)

        total = 0
        while True:
            try:
                result = result_queue.get(timeout=1)
                total += result
                print('Done with a pair.')
                print(f'saved {total} images so far...')
            except queue.Empty:
                if result_queue.empty():
                    break


if __name__ == '__main__':
    main()
