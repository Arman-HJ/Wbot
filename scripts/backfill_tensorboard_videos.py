"""Backfill training mp4s into TensorBoard at the step encoded in the filename.

Usage:
  python scripts/backfill_tensorboard_videos.py \\
    logs/rsl_rl/pnd_adam_pro_velocity_v3_legs/2026-06-09_09-44-13
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rsl_rl.utils.tensorboard_utils import TensorboardSummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Run directory containing events.out.tfevents.* and videos/train/*.mp4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Cap frames per GIF (match runner tensorboard_video_max_frames)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Not a directory: {log_dir}")

    videos = sorted(log_dir.rglob("videos/train/rl-video-step-*.mp4"))
    if not videos:
        raise SystemExit(f"No rl-video-step-*.mp4 under {log_dir}")

    writer = TensorboardSummaryWriter(
        str(log_dir),
        flush_secs=1,
        video_max_frames=args.max_frames,
    )
    writer.logged_videos.clear()

    for video in videos:
        step = int(video.stem.rsplit("-", 1)[-1])
        writer.save_video(video, it=step)
        print(f"uploaded {video.relative_to(log_dir)} at step {step}")

    writer.flush()
    writer.close()
    print(f"Done. Open TensorBoard on {log_dir} and check the IMAGES tab.")


if __name__ == "__main__":
    main()
