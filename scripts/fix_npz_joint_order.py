#!/usr/bin/env python3
"""Reorder ``joint_pos``/``joint_vel`` columns in legacy AMP npz files.

Old ``pkl_to_npz`` logged ``robot.data.joint_pos[0, :]`` (MuJoCo DFS / sim tensor
order) while ``joint_names`` followed ``actuator/motor`` order.  Legs 0–14 match;
indices 15–30 swap head vs arms.

This script assumes stored columns follow MuJoCo compiled hinge order and
rewrites them to match ``joint_names`` (motor) metadata.  Run ``play_npz.py``
with default ``--columns by_name`` afterward to verify.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_XML = (
  _REPO_ROOT / "src/assets/robots/agibot_x2/x2_assets/x2_ultra_simple.xml"
)


def _mjcf_hinge_names(model: mujoco.MjModel) -> list[str]:
  names: list[str] = []
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name:
      names.append(name.split("/")[-1])
  return names


def _load_model(xml_path: Path) -> mujoco.MjModel:
  asset_dir = xml_path.parent
  import os

  orig = Path.cwd()
  try:
    os.chdir(asset_dir)
    return mujoco.MjModel.from_xml_path(xml_path.name)
  finally:
    os.chdir(orig)


def fix_npz_arrays(
  joint_names: list[str],
  joint_pos: np.ndarray,
  joint_vel: np.ndarray | None,
  mjcf_names: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
  if len(joint_names) != len(mjcf_names):
    raise ValueError(
      f"joint_names ({len(joint_names)}) != MuJoCo hinges ({len(mjcf_names)})"
    )
  if joint_names == mjcf_names:
    print("Columns already match MuJoCo hinge order; no permutation applied.")
    return joint_pos, joint_vel

  src_by_name = {n: i for i, n in enumerate(mjcf_names)}
  perm = [src_by_name[n] for n in joint_names]
  fixed_pos = joint_pos[:, perm]
  fixed_vel = joint_vel[:, perm] if joint_vel is not None else None
  return fixed_pos, fixed_vel


def fix_file(
  npz_path: Path,
  xml_path: Path,
  inplace: bool,
  backup: bool,
) -> Path:
  model = _load_model(xml_path)
  mjcf_names = _mjcf_hinge_names(model)

  z = dict(np.load(npz_path, allow_pickle=True))
  joint_names = [str(x) for x in z["joint_names"].tolist()]
  joint_pos = np.asarray(z["joint_pos"], dtype=np.float64)
  joint_vel = (
    np.asarray(z["joint_vel"], dtype=np.float64) if "joint_vel" in z else None
  )

  fixed_pos, fixed_vel = fix_npz_arrays(
    joint_names, joint_pos, joint_vel, mjcf_names
  )
  z["joint_pos"] = fixed_pos
  if fixed_vel is not None:
    z["joint_vel"] = fixed_vel

  out_path = npz_path if inplace else npz_path.with_name(
    npz_path.stem + "_fixed.npz"
  )
  if inplace and backup:
    bak = npz_path.with_suffix(npz_path.suffix + ".bak")
    shutil.copy2(npz_path, bak)
    print(f"Backup: {bak}")

  np.savez(out_path, **z)
  print(f"Wrote: {out_path}")
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser(description="Fix legacy npz joint column order")
  parser.add_argument("npz", type=Path, nargs="+", help="npz file(s) to fix")
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument(
    "--inplace",
    action="store_true",
    help="Overwrite input npz (default: write *_fixed.npz)",
  )
  parser.add_argument(
    "--backup",
    action="store_true",
    help="With --inplace, save .npz.bak before overwrite",
  )
  args = parser.parse_args()

  for path in args.npz:
    fix_file(path.expanduser().resolve(), args.xml.resolve(), args.inplace, args.backup)


if __name__ == "__main__":
  main()
