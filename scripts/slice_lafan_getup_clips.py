#!/usr/bin/env python3
"""Slice supermimic WBOT1026 LAFAN fallAndGetUp CSVs into short AMP get-up clips.

Source (30 Hz, no header, 27 cols = root xyz + quat xyzw + 20 joints)::

  supermimic_wbot_short/data/lafan1_wbot1026_stock_20261105/fallAndGetUp*.csv

Frame ranges match legged_lab ``*_start_end.pkl`` names at the same 30 Hz
source rate, so indices are used directly (no 30→50 remap).

Usage::

  python scripts/slice_lafan_getup_clips.py
  python scripts/slice_lafan_getup_clips.py --enabled-only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = Path(
  "/home/HRJ/workplace/supermimic_wbot_short/data/lafan1_wbot1026_stock_20261105"
)
_DEFAULT_OUT = (
  _REPO
  / "src/assets/amp_motions/wbot_20dof/getup/lafan1/fallAndGetUp_slices"
)
_DEFAULT_XML = _REPO / "src/assets/robots/wbot/xmls/wbot_20dof.xml"

_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)
_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
)

# (clip_stem, start_30, end_30 exclusive, enabled_in_legged_lab)
_SEGMENTS: tuple[tuple[str, int, int, bool], ...] = (
  ("fallAndGetUp1_subject1", 680, 800, False),
  ("fallAndGetUp1_subject1", 850, 940, False),
  ("fallAndGetUp1_subject1", 1060, 1150, True),
  ("fallAndGetUp1_subject1", 1400, 1480, True),
  ("fallAndGetUp1_subject1", 1600, 1700, False),
  ("fallAndGetUp1_subject1", 2100, 2200, True),
  ("fallAndGetUp1_subject1", 2300, 2400, False),
  ("fallAndGetUp1_subject4", 3700, 3800, False),
  ("fallAndGetUp1_subject5", 2100, 2200, False),
  ("fallAndGetUp1_subject5", 2500, 2600, True),
  ("fallAndGetUp1_subject5", 3900, 3980, False),
  ("fallAndGetUp2_subject2", 360, 580, False),
  ("fallAndGetUp2_subject2", 850, 1050, True),
  ("fallAndGetUp2_subject2", 1200, 1370, False),
  ("fallAndGetUp2_subject2", 1500, 1600, False),
  ("fallAndGetUp2_subject3", 450, 550, False),
  ("fallAndGetUp2_subject3", 900, 1000, True),
  ("fallAndGetUp2_subject3", 1850, 1920, False),
  ("fallAndGetUp2_subject3", 2080, 2180, False),
)

_SRC_FPS = 30.0


def _root_ang_vel_world(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
  from scipy.spatial.transform import Rotation as R

  rot = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
  step = (rot[1:] * rot[:-1].inv()).as_rotvec() * fps
  out = np.zeros_like(quat_wxyz[:, :3])
  out[0] = step[0]
  out[-1] = step[-1]
  out[1:-1] = 0.5 * (step[:-1] + step[1:])
  return out


def _csv_to_amp(
  rows: np.ndarray,
  *,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qadr: np.ndarray,
  key_body_ids: list[int],
  fps: float,
) -> dict[str, np.ndarray]:
  if rows.ndim != 2 or rows.shape[1] != 27:
    raise ValueError(f"expected (T, 27) CSV rows, got {rows.shape}")
  if rows.shape[1] - 7 != len(_JOINT_NAMES):
    raise ValueError("joint width mismatch")

  root_pos = rows[:, :3].astype(np.float64)
  # CSV stores xyzw; MuJoCo / AMP use wxyz.
  quat = rows[:, 3:7][:, [3, 0, 1, 2]].astype(np.float64)
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  for t in range(1, len(quat)):
    if np.dot(quat[t - 1], quat[t]) < 0.0:
      quat[t] = -quat[t]
  joint_pos = rows[:, 7:].astype(np.float64)

  dt = 1.0 / fps
  edge = 2 if len(joint_pos) > 2 else 1
  joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=edge)
  root_lin = np.gradient(root_pos, dt, axis=0, edge_order=edge)
  root_ang = _root_ang_vel_world(quat, fps)

  n = len(joint_pos)
  key_pos = np.zeros((n, len(key_body_ids), 3), dtype=np.float64)
  key_quat = np.zeros((n, len(key_body_ids), 4), dtype=np.float64)
  local = np.zeros(3)
  conj = np.zeros(4)
  for t in range(n):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = quat[t]
    data.qpos[joint_qadr] = joint_pos[t]
    data.qvel[0:3] = root_lin[t]
    mujoco.mju_negQuat(conj, quat[t])
    mujoco.mju_rotVecQuat(local, root_ang[t], conj)
    data.qvel[3:6] = local
    mujoco.mj_forward(model, data)
    for i, bid in enumerate(key_body_ids):
      key_pos[t, i] = data.xpos[bid]
      key_quat[t, i] = data.xquat[bid]

  return {
    "fps": np.asarray([fps], dtype=np.float64),
    "joint_names": np.asarray(_JOINT_NAMES),
    "key_body_names": np.asarray(_KEY_BODY_NAMES),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": root_pos.reshape(n, 1, 3).astype(np.float32),
    "body_quat_w": quat.reshape(n, 1, 4).astype(np.float32),
    "body_lin_vel_w": root_lin.reshape(n, 1, 3).astype(np.float32),
    "body_ang_vel_w": root_ang.reshape(n, 1, 3).astype(np.float32),
    "key_body_pos": key_pos.reshape(n, -1).astype(np.float32),
    "key_body_quat_w": key_quat.astype(np.float32),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--src-dir", type=Path, default=_DEFAULT_SRC)
  parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument(
    "--enabled-only",
    action="store_true",
    help="Only write clips that legged_lab trains with (weight=1).",
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete out-dir before writing.",
  )
  args = parser.parse_args()

  src_dir = args.src_dir.resolve()
  out_dir = args.out_dir.resolve()
  if args.clean and out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
  data = mujoco.MjData(model)
  joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _JOINT_NAMES
  ]
  if min(joint_ids) < 0:
    missing = [n for n, i in zip(_JOINT_NAMES, joint_ids) if i < 0]
    raise KeyError(f"joints missing from XML: {missing}")
  joint_qadr = model.jnt_qposadr[joint_ids]
  key_body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _KEY_BODY_NAMES
  ]
  if min(key_body_ids) < 0:
    missing = [n for n, i in zip(_KEY_BODY_NAMES, key_body_ids) if i < 0]
    raise KeyError(f"key bodies missing from XML: {missing}")

  cache: dict[str, np.ndarray] = {}
  written: list[Path] = []

  for stem, start, end, enabled in _SEGMENTS:
    if args.enabled_only and not enabled:
      continue
    src_path = src_dir / f"{stem}.csv"
    if not src_path.is_file():
      print(f"[skip] missing source {src_path}")
      continue
    if stem not in cache:
      cache[stem] = np.loadtxt(src_path, delimiter=",")
      if cache[stem].ndim != 2 or cache[stem].shape[1] != 27:
        raise ValueError(f"{src_path.name}: bad shape {cache[stem].shape}")
    rows = cache[stem]
    n = rows.shape[0]
    if end > n:
      print(f"[skip] {stem}_{start}_{end}: [{start}:{end}] > T={n}")
      continue
    if end <= start:
      print(f"[skip] empty range {stem}_{start}_{end}")
      continue

    sliced = rows[start:end]
    amp = _csv_to_amp(
      sliced,
      model=model,
      data=data,
      joint_qadr=joint_qadr,
      key_body_ids=key_body_ids,
      fps=_SRC_FPS,
    )
    root_z = amp["body_pos_w"][:, 0, 2]
    out_name = f"{stem}_{start}_{end}.npz"
    out_path = out_dir / out_name
    np.savez_compressed(out_path, **amp)
    # Also keep the raw CSV slice for supermimic-side replay.
    np.savetxt(out_dir / f"{stem}_{start}_{end}.csv", sliced, delimiter=",", fmt="%.6f")
    written.append(out_path)
    tag = "enabled" if enabled else "extra"
    print(
      f"[ok] {out_name:48s}  {tag:7s}  "
      f"src30=[{start}:{end}]  T={end - start}  "
      f"dur={(end - start - 1) / _SRC_FPS:.2f}s  "
      f"root_z=[{root_z.min():.2f},{root_z.max():.2f}]"
    )

  print(f"\nWrote {len(written)} clips → {out_dir}")
  print(
    "Source:",
    src_dir,
    "(30 Hz CSV; frame indices match legged_lab names directly)",
  )
  print(
    "Note: fallAndGetUp6_* is not in this pack; those legged_lab clips "
    "cannot be sliced here."
  )


if __name__ == "__main__":
  main()
