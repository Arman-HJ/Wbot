#!/usr/bin/env python3
"""Convert HumanoidSoccer wbot soccer-standard NPZs to AmpLSMotionLoader format."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES

_DEFAULT_SRC = Path("/home/HRJ/workplace/HumanoidSoccer/motions/wbot-soccer-standard")
_DEFAULT_OUT = Path("src/assets/amp_motions/wbot_20dof/soccer_standard")


def _root_ang_vel_w(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
  T = len(quat_wxyz)
  q = quat_wxyz.astype(np.float64).copy()
  for t in range(1, T):
    if np.dot(q[t - 1], q[t]) < 0.0:
      q[t] = -q[t]
  dt = 1.0 / float(fps)
  ang = np.zeros((T, 3), dtype=np.float64)
  for t in range(T):
    t0 = max(0, t - 1)
    t1 = min(T - 1, t + 1)
    w0, x0, y0, z0 = q[t0]
    w1, x1, y1, z1 = q[t1]
    cw, cx, cy, cz = w0, -x0, -y0, -z0
    rw = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    rx = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    ry = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    rz = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    span = (t1 - t0) * dt
    if span <= 0:
      continue
    if rw < 0:
      rw, rx, ry, rz = -rw, -rx, -ry, -rz
    sin_half = np.sqrt(max(0.0, 1.0 - rw * rw))
    if sin_half < 1e-8:
      continue
    angle = 2.0 * np.arctan2(sin_half, rw)
    axis = np.array([rx, ry, rz]) / sin_half
    ang[t] = axis * (angle / span)
  return ang


def convert_one(src: Path, out_dir: Path) -> Path:
  d = np.load(src, allow_pickle=True)
  src_joints = [str(x) for x in d["joint_names"]]
  src_bodies = [str(x) for x in d["body_names"]]
  j_idx = [src_joints.index(n) for n in WBOT_JOINT_NAMES]
  b_idx = [src_bodies.index(n) for n in WBOT_KEY_BODY_NAMES]
  pelvis_i = src_bodies.index("pelvis")
  fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  jp = np.asarray(d["joint_pos"], dtype=np.float32)[:, j_idx]
  jv = np.asarray(d["joint_vel"], dtype=np.float32)[:, j_idx]
  root_pos = np.asarray(d["body_pos_w"][:, pelvis_i], dtype=np.float32)
  root_quat = np.asarray(d["body_quat_w"][:, pelvis_i], dtype=np.float32)
  root_lin = np.asarray(d["body_lin_vel_w"][:, pelvis_i], dtype=np.float32)
  root_ang = np.asarray(d["body_ang_vel_w"][:, pelvis_i], dtype=np.float32)
  if not np.any(np.abs(root_ang) > 1e-8):
    root_ang = _root_ang_vel_w(root_quat, fps).astype(np.float32)
  key_pos = np.asarray(d["body_pos_w"][:, b_idx], dtype=np.float32)
  key_quat = np.asarray(d["body_quat_w"][:, b_idx], dtype=np.float32)
  out_path = out_dir / src.name.replace("_wbot.npz", ".npz")
  np.savez_compressed(
    out_path,
    fps=np.asarray(fps, dtype=np.float64),
    joint_names=np.asarray(WBOT_JOINT_NAMES),
    key_body_names=np.asarray(WBOT_KEY_BODY_NAMES),
    joint_pos=jp,
    joint_vel=jv,
    body_pos_w=root_pos.reshape(-1, 1, 3),
    body_quat_w=root_quat.reshape(-1, 1, 4),
    body_lin_vel_w=root_lin.reshape(-1, 1, 3),
    body_ang_vel_w=root_ang.reshape(-1, 1, 3),
    key_body_pos=key_pos.reshape(len(jp), -1),
    key_body_quat_w=key_quat,
    source_npz=np.asarray(str(src)),
  )
  return out_path


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--src-dir", type=Path, default=_DEFAULT_SRC)
  ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  paths = sorted(args.src_dir.glob("*_wbot.npz"))
  if not paths:
    raise SystemExit(f"No *_wbot.npz under {args.src_dir}")
  for src in paths:
    out = convert_one(src, args.out_dir)
    print(out)
  print(f"wrote {len(paths)} files -> {args.out_dir}")


if __name__ == "__main__":
  main()
