import torchvision
import torch
import argparse
import toml
from tqdm import tqdm


def process_pair(l, r, step, offset, root):
    l_video, _, _ = torchvision.io.video.read_video(
        l, pts_unit="sec", output_format="TCHW"
    )
    r_video, _, _ = torchvision.io.video.read_video(
        r, pts_unit="sec", output_format="TCHW"
    )

    l_len = l_video.shape[0]
    r_len = r_video.shape[0]

    shorter = l_len if l_len < r_len else r_len

    r_offset = offset + r_len

    i = 0
    while i < shorter:
        l_frame = l_video[i]
        r_frame = r_video[i]
        combined = torch.cat([l_frame, r_frame], dim=-1)
        fname = f"{root}/{i+offset}_{i+r_offset}"
        torchvision.io.write_jpeg(combined, fname, quality=100)
        i += step

    return l_len


def main():
    parser = argparse.ArgumentParser(description="train the timescale diffusion model")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("output_path", help="run name.")
    parser.add_argument("--step", help="step size", type=int, default=100)
    parser.add_argument("--offset", help="frame offset", type=int, default=0)
    args = parser.parse_args()

    config = toml.decoder.load(args.config_file)

    lhs_vids = config["lhs_videos"]
    rhs_vids = config["rhs_videos"]

    assert len(lhs_vids) == len(rhs_vids)

    off = args.offset
    for i in tqdm(range(len(lhs_vids))):
        length = process_pair(
            lhs_vids[i], rhs_vids[i], args.step, off, args.output_path
        )
        off += length


if __name__ == "__main__":
    main()
