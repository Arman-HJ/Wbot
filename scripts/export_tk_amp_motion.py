#!/usr/bin/env python3
"""Export mjlab AMP ``.npz`` clips to TienKung ``AMPLoader`` JSON (``.txt``).

Example::

  python scripts/export_tk_amp_motion.py \\
    --npz src/assets/amp_motions/pnd_adam_sp/run3/forward_walk_woman_pnd_adam_sp.npz \\
    --out src/assets/amp_motions/pnd_adam_sp/tk_amp/walk.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.tk_amp_obs import (  # noqa: E402
  AMP_OBS_DIM,
  FOOT_BODIES,
  HAND_BODIES,
  LEFT_ARM_JOINTS,
  LEFT_LEG_JOINTS,
  RIGHT_ARM_JOINTS,
  RIGHT_LEG_JOINTS,
  WAIST_JOINTS,
)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vectors ``v`` by quaternions ``q`` (wxyz)."""
  w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
  qvec = q[:, 1:]
  uv = np.cross(qvec, v)
  uuv = np.cross(qvec, uv)
  return v + 2 * (w * uv + uuv)


def _quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  q_inv = q.copy()
  q_inv[:, 1:] *= -1.0
  return _quat_apply(q_inv, v)


def _index_map(names: list[str], keys: tuple[str, ...]) -> list[int]:
  lookup = {n: i for i, n in enumerate(names)}
  missing = [k for k in keys if k not in lookup]
  if missing:
    raise KeyError(f"Missing names in NPZ: {missing}")
  return [lookup[k] for k in keys]


def npz_to_amp_frames(npz_path: Path) -> tuple[list[list[float]], float]:
  z = np.load(npz_path, allow_pickle=True)
  fps = float(np.asarray(z["fps"]).reshape(-1)[0])
  frame_dt = 1.0 / fps
  joint_names = [str(n) for n in z["joint_names"].tolist()]
  body_names = [str(n) for n in z["body_names"].tolist()]
  jp = np.asarray(z["joint_pos"], dtype=np.float64)
  jv = np.asarray(z["joint_vel"], dtype=np.float64)
  bp = np.asarray(z["body_pos_w"], dtype=np.float64)
  bq = np.asarray(z["body_quat_w"], dtype=np.float64)

  ra = _index_map(joint_names, RIGHT_ARM_JOINTS)
  la = _index_map(joint_names, LEFT_ARM_JOINTS)
  rl = _index_map(joint_names, RIGHT_LEG_JOINTS)
  ll = _index_map(joint_names, LEFT_LEG_JOINTS)
  wa = _index_map(joint_names, WAIST_JOINTS)
  hands = _index_map(body_names, HAND_BODIES)
  feet = _index_map(body_names, FOOT_BODIES)
  pelvis = body_names.index("pelvis")

  root_pos = bp[:, pelvis, :]
  root_quat = bq[:, pelvis, :]
  left_offset = np.array([0.0, 0.0, -0.25], dtype=np.float64)
  right_offset = np.array([0.0, 0.0, -0.25], dtype=np.float64)

  left_hand = (
    bp[:, hands[0], :]
    - root_pos
    + _quat_apply(bq[:, hands[0], :], np.broadcast_to(left_offset, root_pos.shape))
  )
  right_hand = (
    bp[:, hands[1], :]
    - root_pos
    + _quat_apply(bq[:, hands[1], :], np.broadcast_to(right_offset, root_pos.shape))
  )
  left_hand = _quat_apply_inverse(root_quat, left_hand)
  right_hand = _quat_apply_inverse(root_quat, right_hand)
  left_foot = _quat_apply_inverse(root_quat, bp[:, feet[0], :] - root_pos)
  right_foot = _quat_apply_inverse(root_quat, bp[:, feet[1], :] - root_pos)

  frames = np.concatenate(
    [
      jp[:, ra],
      jp[:, la],
      jp[:, rl],
      jp[:, ll],
      jp[:, wa],
      jv[:, ra],
      jv[:, la],
      jv[:, rl],
      jv[:, ll],
      jv[:, wa],
      left_hand,
      right_hand,
      left_foot,
      right_foot,
    ],
    axis=-1,
  )
  if frames.shape[1] != AMP_OBS_DIM:
    raise RuntimeError(f"Expected AMP dim {AMP_OBS_DIM}, got {frames.shape[1]}")
  return frames.tolist(), frame_dt


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--npz", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--motion-weight", type=float, default=0.5)
  args = parser.parse_args()

  frames, frame_dt = npz_to_amp_frames(args.npz)
  doc = {
    "LoopMode": "Wrap",
    "FrameDuration": frame_dt,
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": True,
    "MotionWeight": args.motion_weight,
    "Frames": frames,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("w", encoding="utf-8") as f:
    json.dump(doc, f, indent=None)
    # Pretty-print header style similar to TienKung walk.txt
  # Rewrite with readable framing.
  with args.out.open("w", encoding="utf-8") as f:
    f.write("{\n")
    f.write('"LoopMode": "Wrap",\n')
    f.write(f'"FrameDuration": {frame_dt:.6g},\n')
    f.write('"EnableCycleOffsetPosition": true,\n')
    f.write('"EnableCycleOffsetRotation": true,\n')
    f.write(f'"MotionWeight": {args.motion_weight},\n\n')
    f.write('"Frames":\n[\n')
    for i, row in enumerate(frames):
      line = "  [" + ", ".join(f"{x:.6f}" for x in row) + "]"
      if i + 1 < len(frames):
        line += ","
      f.write(line + "\n")
    f.write("]\n}\n")
  print(f"Wrote {len(frames)} frames ({AMP_OBS_DIM} dim) -> {args.out}")


if __name__ == "__main__":
  main()
