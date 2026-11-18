#!/usr/bin/env python3
"""Play TienKung-style AMP ``.txt`` (disc obs frames) in MuJoCo.

``walk.txt`` stores per-frame AMP observations::
  [right_arm(7), left_arm(7), right_leg(6), left_leg(6), waist(3),
   same-order joint_vel(29), left/right hand+foot EE in root (12)]
  = 70 dims.  Root pose is not stored; pelvis is held fixed for visualization.

Example::

  python scripts/play_tk_amp_txt.py \\
    --txt src/assets/amp_motions/pnd_adam_sp/tk_amp/walk.txt
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.tk_amp_obs import (  # noqa: E402
  AMP_JOINT_POS_SIZE,
  LEFT_ARM_JOINTS,
  LEFT_LEG_JOINTS,
  RIGHT_ARM_JOINTS,
  RIGHT_LEG_JOINTS,
  WAIST_JOINTS,
)

_DEFAULT_TXT = (
  _REPO_ROOT / "src/assets/amp_motions/pnd_adam_sp/tk_amp/walk.txt"
)
_DEFAULT_XML = _REPO_ROOT / "src/assets/robots/adam_sp/xml/adam_sp.xml"

# Same block order as ``export_tk_amp_motion.npz_to_amp_frames``.
_AMP_JOINT_ORDER = (
  RIGHT_ARM_JOINTS
  + LEFT_ARM_JOINTS
  + RIGHT_LEG_JOINTS
  + LEFT_LEG_JOINTS
  + WAIST_JOINTS
)


def _load_tk_amp_txt(path: Path) -> tuple[np.ndarray, float]:
  text = path.read_text(encoding="utf-8")
  dt_m = re.search(r'"FrameDuration"\s*:\s*([0-9.eE+-]+)', text)
  if not dt_m:
    raise ValueError(f"FrameDuration not found in {path}")
  frame_dt = float(dt_m.group(1))
  frames_m = re.search(r'"Frames"\s*:\s*(\[[\s\S]*\])\s*\}?\s*$', text)
  if not frames_m:
    raise ValueError(f"Frames array not found in {path}")
  frames = np.asarray(ast.literal_eval(frames_m.group(1)), dtype=np.float64)
  if frames.ndim != 2 or frames.shape[1] < AMP_JOINT_POS_SIZE:
    raise ValueError(
      f"Expected (T, >= {AMP_JOINT_POS_SIZE}) frames, got {frames.shape}"
    )
  return frames, frame_dt


def _joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
  out: dict[str, int] = {}
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None:
      continue
    adr = int(model.jnt_qposadr[j])
    out[name] = adr
    out[name.split("/")[-1]] = adr
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--txt", type=Path, default=_DEFAULT_TXT)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--loop", action="store_true", default=True)
  parser.add_argument("--no-loop", action="store_false", dest="loop")
  parser.add_argument(
    "--root-z",
    type=float,
    default=0.92,
    help="Fixed free-joint height (AMP txt has no root trajectory).",
  )
  args = parser.parse_args()

  frames, frame_dt = _load_tk_amp_txt(args.txt)
  joint_pos = frames[:, :AMP_JOINT_POS_SIZE]
  assert len(_AMP_JOINT_ORDER) == AMP_JOINT_POS_SIZE

  model = mujoco.MjModel.from_xml_path(str(args.xml))
  data = mujoco.MjData(model)
  qmap = _joint_qpos_map(model)
  missing = [n for n in _AMP_JOINT_ORDER if n not in qmap]
  if missing:
    raise KeyError(f"Joints missing in MJCF: {missing}")

  has_free = int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
  dt = frame_dt / max(args.speed, 1e-6)
  n = joint_pos.shape[0]
  print(
    f"Playing {args.txt.name}: {n} frames @ {1.0 / frame_dt:.1f} fps "
    f"(dt={frame_dt:.4f}s), speed={args.speed}, xml={args.xml}",
    flush=True,
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    i = 0
    while viewer.is_running():
      t0 = time.perf_counter()
      data.qpos[:] = 0.0
      data.qvel[:] = 0.0
      if has_free:
        data.qpos[0:3] = [0.0, 0.0, args.root_z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
      for j, name in enumerate(_AMP_JOINT_ORDER):
        data.qpos[qmap[name]] = joint_pos[i, j]
      mujoco.mj_forward(model, data)
      viewer.sync()
      i += 1
      if i >= n:
        if not args.loop:
          break
        i = 0
      elapsed = time.perf_counter() - t0
      time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
  main()
