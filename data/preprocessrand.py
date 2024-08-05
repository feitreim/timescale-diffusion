import torchvision
import torch
import queue
import argparse
import os
import toml
import random
import multiprocessing as mp
import cv2

from pathlib import Path


def get_frame_count(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return -1

    # Get the total number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video object
    video.release()

    return frame_count


def process_pair(l, r, max_distance, step, offset, root):
    print("processing pair.")
    l_video = torchvision.io.VideoReader(l, "video")
    l_video.set_current_stream("video:0")
    print("l_video open")
    r_video = torchvision.io.VideoReader(r, "video")
    print("r_video open")
    r_video.set_current_stream("video:0")

    l_len = get_frame_count(l)
    fps = 30
    r_len = get_frame_count(r)
    t_len = int(l_len + r_len)

    sub_dir = Path(root) / str(offset)
    os.makedirs(sub_dir, exist_ok=True)

    i, total = 0, 0
    while i < l_len and i + max_distance < t_len:
        distance = random.randint(0, max_distance)
        if i + distance >= l_len:
            l_video.seek(i / fps)
            l_frame = next(l_video)["data"]
            idx = (i + distance - l_len) / fps
            idx = idx if idx > 0 else 1
            r_video.seek(idx)
            r_frame = next(r_video)["data"]
        else:
            l_video.seek(i / fps)
            l_frame = next(l_video)["data"]
            l_video.seek((i + distance) / fps)
            r_frame = next(l_video)["data"]
        combined = torch.cat([l_frame, r_frame], dim=-1)
        fname = sub_dir / f"{offset+i}_{offset+i+distance}"
        torchvision.io.write_jpeg(combined, fname, quality=100)
        i += step + random.randint(-step // 2, step // 2)
        total += 1
        print(f"saved file {fname}. this proc saved {total}")

    return total


def compute_frame_timestamps(first_vid, video_file_paths):
    """
    Compute proper timestamps for each video from oldest->newest.
    Pre:
        - Videos should already be in order.
        - Videos should contain timestamp information in YYMMDD_HHMMSS
    """

    def compute_date_info(path):
        filename = path.split("/")[-1]
        filename = filename.casefold().strip("adcdefghijklmnopqrstuvwxyz,.;'[]{}:<>?/")
        yearmonthday = filename.split("_")[0]
        hourminsec = filename.split("_")[-1]

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
    parser = argparse.ArgumentParser(description="train the timescale diffusion model")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("output_path", help="run name.")
    parser.add_argument("--num_shards", help="total number of shards", type=int)
    parser.add_argument("--shard", help="which shard this is.", type=int)
    parser.add_argument("--step", help="step size", type=int, default=100)
    parser.add_argument(
        "--distance", help="max distance between frames", type=int, default=100000
    )
    args = parser.parse_args()
    config = toml.decoder.load(args.config_file)

    lhs_vids = config["lhs_videos"]
    rhs_vids = config["rhs_videos"]
    offsets = compute_frame_timestamps(lhs_vids[0], lhs_vids)
    assert len(lhs_vids) == len(rhs_vids)

    # sharding
    shard_size = len(lhs_vids) // args.num_shards
    start = shard_size * args.shard
    end = len(lhs_vids)
    if not args.shard == args.num_shards - 1:
        end = shard_size * (args.shard + 1)

    lhs_vids = lhs_vids[start:end]
    rhs_vids = rhs_vids[start:end]
    offsets = offsets[start:end]

    print(f"shard size: {shard_size}, s: {start}, e: {end}")

    with mp.Pool(8) as pool:
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
                print("Done with a pair.")
                print(f"saved {total} images so far...")
            except queue.Empty:
                if result_queue.empty():
                    break


if __name__ == "__main__":
    main()
