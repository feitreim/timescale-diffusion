import torchvision
import torch
import argparse
import toml
import random
from tqdm import tqdm


def process_pair(l, r, max_distance, step, offset, root):
    def get_distances(max):
        distances = []
        d = max
        while d > 0:
            distances.append(d)
            d = d // 2
        return distances

    distances = get_distances(max_distance)
    l_video, _, _ = torchvision.io.video.read_video(
        l, pts_unit="sec", output_format="TCHW"
    )
    r_video, _, _ = torchvision.io.video.read_video(
        r, pts_unit="sec", output_format="TCHW"
    )

    l_len = l_video.shape[0]

    t_video = torch.cat([l_video, r_video], dim=0)
    t_len = t_video.shape[0]

    i = 0
    while i < l_len and i + max_distance < t_len:
        distance = distances[random.randint(0, len(distances))]
        l_frame = t_video[i]
        r_frame = t_video[i + distance]
        combined = torch.cat([l_frame, r_frame], dim=-1)
        fname = f"{root}/{i+offset}_{i+distance}"
        torchvision.io.write_jpeg(combined, fname, quality=100)
        i += step + random.randint(-step//2, step//2)

    return l_len


def compute_frame_timestamps(first_vid, video_file_paths):
    """
    Compute proper timestamps for each video from oldest->newest.
    Pre:
        - Videos should already be in order.
        - Videos should contain timestamp information in YYMMDD_HHMMSS
    """
    def compute_date_info(path):
        filename = path.split("/")[-1]
        filename = filename.casefold().strip(
            "adcdefghijklmnopqrstuvwxyz,.;'[]{}:<>?/"
        )
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

def main():
    parser = argparse.ArgumentParser(description="train the timescale diffusion model")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("output_path", help="run name.")
    parser.add_argument("--step", help="step size", type=int, default=100)
    parser.add_argument("--distance", help="max distance between frames", type=int, default = 100000)
    args = parser.parse_args()

    config = toml.decoder.load(args.config_file)

    lhs_vids = config["lhs_videos"]
    rhs_vids = config["rhs_videos"]

    offsets = compute_frame_timestamps(lhs_vids[0], lhs_vids)

    assert len(lhs_vids) == len(rhs_vids)

    for i in tqdm(range(len(lhs_vids))):
        process_pair(
            lhs_vids[i], rhs_vids[i], args.distance, args.step, offsets[i], args.output_path
        )


if __name__ == "__main__":
    main()
