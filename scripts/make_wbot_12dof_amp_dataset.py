"""Derive leg-only WBOT 12-DoF AMP clips from the WBOT 20-DoF AMP clips.

The 12-DoF model shares the 20-DoF leg chain and mass, so the root and leg
trajectories carry over unchanged; arm joints and arm key bodies are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.tasks.amp.wbot_constants import (
  WBOT_12DOF_JOINT_NAMES,
  WBOT_12DOF_KEY_BODY_NAMES,
)

CLIPS = (
  "walk_moving",
  "run",
  "run_mirror",
  "turn_l",
  "turn_r",
  "side_l",
  "side_r",
  "stand",
)


def convert(src: Path, dst: Path) -> None:
  with np.load(src, allow_pickle=False) as data:
    out = {key: data[key] for key in data.files}
  joint_names = out["joint_names"].astype(str).tolist()
  key_names = out["key_body_names"].astype(str).tolist()
  joint_ids = [joint_names.index(name) for name in WBOT_12DOF_JOINT_NAMES]
  key_ids = [key_names.index(name) for name in WBOT_12DOF_KEY_BODY_NAMES]

  num_frames = out["joint_pos"].shape[0]
  key_pos = out["key_body_pos"].reshape(num_frames, len(key_names), 3)
  out["joint_pos"] = out["joint_pos"][:, joint_ids]
  out["joint_vel"] = out["joint_vel"][:, joint_ids]
  out["key_body_pos"] = key_pos[:, key_ids].reshape(num_frames, -1)
  out["key_body_quat_w"] = out["key_body_quat_w"][:, key_ids]
  out["joint_names"] = np.array(WBOT_12DOF_JOINT_NAMES)
  out["key_body_names"] = np.array(WBOT_12DOF_KEY_BODY_NAMES)

  dst.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(dst, **out)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--src",
    type=Path,
    default=Path("src/assets/amp_motions/wbot_20dof/walk_run_amp"),
  )
  parser.add_argument(
    "--dst",
    type=Path,
    default=Path("src/assets/amp_motions/wbot_12dof/walk_run_amp"),
  )
  args = parser.parse_args()
  for clip in CLIPS:
    convert(args.src / f"{clip}.npz", args.dst / f"{clip}.npz")
    print(f"{args.src / clip}.npz -> {args.dst / clip}.npz")


if __name__ == "__main__":
  main()
