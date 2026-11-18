#!/usr/bin/env python3
"""Play mjlab AMP ``.npz`` expert motion in MuJoCo (verify joint mapping / retarget quality).

Usage (from unitree_rl_mjlab repo root)::

  python scripts/play_npz.py \\
      --npz-file src/assets/amp_motions/x2/robot_stand_walk_run/stand-walk_agibot_x2.npz

  # If pkl_to_npz saved sim tensor order but joint_names are motor order:
  python scripts/play_npz.py --npz-file path/to/clip.npz --columns mjcf

Controls: drag orbit; Esc quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_XML = (
  _REPO_ROOT / "src/assets/robots/agibot_x2/x2_assets/scene.xml"
)
_DEFAULT_NPZ = (
  _REPO_ROOT
  / "src/assets/amp_motions/x2/robot_stand_walk_run/stand-walk_agibot_x2.npz"
)


def _build_joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
  out: dict[str, int] = {}
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None:
      continue
    short = name.split("/")[-1]
    adr = int(model.jnt_qposadr[j])
    out[name] = adr
    out[short] = adr
  return out


def _mjcf_hinge_names(model: mujoco.MjModel) -> list[str]:
  names: list[str] = []
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name:
      names.append(name.split("/")[-1])
  return names


def _resolve_qposadr(joint_map: dict[str, int], joint_name: str) -> int:
  if joint_name in joint_map:
    return joint_map[joint_name]
  short = joint_name.split("/")[-1]
  if short in joint_map:
    return joint_map[short]
  raise KeyError(f"Joint {joint_name!r} not found in MuJoCo model")


def _apply_frame(
  data: mujoco.MjData,
  model: mujoco.MjModel,
  joint_map: dict[str, int],
  joint_names: list[str],
  joint_pos: np.ndarray,
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  # free root (pelvis)
  data.qpos[0:3] = root_pos
  data.qpos[3:7] = root_quat_wxyz
  for i, jname in enumerate(joint_names):
    adr = _resolve_qposadr(joint_map, jname)
    data.qpos[adr] = float(joint_pos[i])
  mujoco.mj_forward(model, data)


def _print_order_check(model: mujoco.MjModel, joint_names: list[str]) -> None:
  import xml.etree.ElementTree as ET

  xml = _DEFAULT_XML
  motor = [
    m.get("joint")
    for m in ET.parse(xml).getroot().findall(".//actuator/motor")
    if m.get("joint")
  ]
  hinge: list[str] = []
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name:
      hinge.append(name.split("/")[-1])

  print("=== joint order check ===")
  print(f"npz joint_names ({len(joint_names)}): motor-order metadata")
  print(f"  matches actuator/motor list: {joint_names == motor}")
  print(f"MuJoCo hinge order ({len(hinge)}) DFS/body-tree in compiled model:")
  for i, (n, h) in enumerate(zip(joint_names, hinge)):
    mark = "" if n == h else " <-- MISMATCH"
    if n != h:
      print(f"  {i:2d} npz={n:32s} mjcf={h:32s}{mark}")
  if joint_names == hinge:
    print("All names match MuJoCo hinge order (column index may still be wrong if npz was saved in sim tensor order).")
  print(
    "Playback: --columns by_name  => column i -> joint_names[i];"
    " --columns mjcf => column i -> MuJoCo hinge[i].\n"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Play AMP motion .npz in MuJoCo viewer")
  parser.add_argument("--npz-file", type=Path, default=_DEFAULT_NPZ)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--root-body-index", type=int, default=0)
  parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
  parser.add_argument("--loop", action="store_true", help="Loop motion")
  parser.add_argument(
    "--print-order",
    action="store_true",
    help="Print npz joint_names vs MuJoCo vs motor order, then exit",
  )
  parser.add_argument(
    "--columns",
    choices=("by_name", "mjcf"),
    default="by_name",
    help=(
      "by_name: joint_pos[:,i] is for joint_names[i] (npz metadata). "
      "mjcf: joint_pos[:,i] is for MuJoCo compiled hinge order (Isaac/mjlab sim tensor)."
    ),
  )
  args = parser.parse_args()

  npz_path = args.npz_file.expanduser().resolve()
  xml_path = args.xml.expanduser().resolve()
  if not npz_path.is_file():
    raise FileNotFoundError(npz_path)
  if not xml_path.is_file():
    raise FileNotFoundError(xml_path)

  z = np.load(npz_path, allow_pickle=True)
  fps = float(np.asarray(z["fps"]).reshape(-1)[0])
  joint_names = [str(x) for x in z["joint_names"].tolist()]
  joint_pos = np.asarray(z["joint_pos"], dtype=np.float64)
  body_pos_w = np.asarray(z["body_pos_w"], dtype=np.float64)
  body_quat_w = np.asarray(z["body_quat_w"], dtype=np.float64)
  root_idx = int(z["root_body_index"]) if "root_body_index" in z else args.root_body_index

  if joint_pos.shape[1] != len(joint_names):
    raise ValueError(
      f"joint_pos width {joint_pos.shape[1]} != len(joint_names) {len(joint_names)}"
    )

  # Load MJCF from asset dir so meshdir resolves.
  orig_cwd = Path.cwd()
  asset_dir = xml_path.parent
  try:
    import os

    os.chdir(asset_dir)
    model = mujoco.MjModel.from_xml_path(xml_path.name)
  finally:
    import os

    os.chdir(orig_cwd)

  data = mujoco.MjData(model)
  joint_map = _build_joint_qpos_map(model)
  play_joint_names = list(joint_names)
  if args.columns == "mjcf":
    play_joint_names = _mjcf_hinge_names(model)
    if len(play_joint_names) != len(joint_names):
      raise ValueError(
        f"MuJoCo hinges {len(play_joint_names)} != npz dof {len(joint_names)}"
      )

  if args.print_order:
    _print_order_check(model, joint_names)
    return

  dt = 1.0 / fps / max(args.speed, 1e-6)
  num_frames = joint_pos.shape[0]
  print(f"Playing {npz_path.name}: {num_frames} frames @ {fps} fps (dt={dt:.4f}s)")
  print(f"Model: {xml_path}")
  print(f"Column mapping: {args.columns}")
  if args.columns == "by_name" and joint_names != play_joint_names:
    print(
      "NOTE: joint_names (motor) != MuJoCo hinge order at idx 15–30. "
      "If arms/head look wrong, run: python scripts/fix_npz_joint_order.py <npz> --inplace --backup"
    )
  print("Controls: drag to orbit; Esc close.")

  frame = 0
  paused = False
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      if not paused:
        root_pos = body_pos_w[frame, root_idx, :]
        root_quat = body_quat_w[frame, root_idx, :]
        _apply_frame(
          data,
          model,
          joint_map,
          play_joint_names,
          joint_pos[frame],
          root_pos,
          root_quat,
        )
        viewer.sync()
        frame += 1
        if frame >= num_frames:
          if args.loop:
            frame = 0
          else:
            frame = num_frames - 1
        time.sleep(dt)
      else:
        viewer.sync()
        time.sleep(0.05)


if __name__ == "__main__":
  main()
