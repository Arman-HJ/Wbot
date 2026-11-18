#!/usr/bin/env python3
"""Render AMP motion ``.npz`` clips to MP4 (kinematic replay).

Usage::

  python scripts/render_amp_motion_videos.py \\
    --motion-dir src/assets/amp_motions/wbot_20dof/walk_run_amp

Videos are written to ``<motion-dir>/video/<stem>.mp4`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gpu_gl import use_gpu_egl

use_gpu_egl()

import imageio.v2 as imageio
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_XML = (
  _REPO / "src/assets/robots/wbot/xmls/wbot_20dof.xml"
)


def _build_model(xml_path: Path, width: int, height: int) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.visual.global_.offwidth = max(width, 640)
  spec.visual.global_.offheight = max(height, 480)
  world = spec.worldbody
  world.add_light(pos=(2.0, -2.0, 3.0), dir=(-0.5, 0.5, -1.0))
  world.add_light(pos=(-2.0, 2.0, 3.0), dir=(0.5, -0.5, -1.0))
  floor = world.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = (5.0, 5.0, 0.01)
  floor.rgba = (0.35, 0.35, 0.38, 1.0)
  return spec.compile()


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


def _apply_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_map: dict[str, int],
  joint_names: list[str],
  joint_pos: np.ndarray,
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  if model.nq >= 7:
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
  for i, name in enumerate(joint_names):
    adr = joint_map.get(name)
    if adr is not None:
      data.qpos[adr] = float(joint_pos[i])
  mujoco.mj_forward(model, data)


def render_npz(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_map: dict[str, int],
  npz_path: Path,
  out_path: Path,
  *,
  width: int,
  height: int,
  camera_distance: float,
) -> None:
  motion = np.load(npz_path, allow_pickle=True)
  joint_names = [str(n) for n in motion["joint_names"]]
  joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
  body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
  body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
  fps = float(np.asarray(motion["fps"]).ravel()[0])
  n = joint_pos.shape[0]

  renderer = mujoco.Renderer(model, height=height, width=width)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = camera_distance
  cam.azimuth = 135.0
  cam.elevation = -15.0
  scene_option = mujoco.MjvOption()

  out_path.parent.mkdir(parents=True, exist_ok=True)
  frames: list[np.ndarray] = []
  for t in range(n):
    root = body_pos[t, 0]
    _apply_frame(
      model,
      data,
      joint_map,
      joint_names,
      joint_pos[t],
      root,
      body_quat[t, 0],
    )
    cam.lookat[:] = (float(root[0]), float(root[1]), float(root[2]) + 0.15)
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    frames.append(renderer.render().copy())

  imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
  renderer.close()
  print(
    f"[video] {out_path.name}  frames={n} fps={fps:.0f} "
    f"dur={n / max(fps, 1e-6):.1f}s → {out_path}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion-dir",
    type=Path,
    default=_REPO / "src/assets/amp_motions/wbot_20dof/walk_run_amp",
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Default: <motion-dir>/video",
  )
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--distance", type=float, default=2.4)
  parser.add_argument(
    "--glob",
    default="*.npz",
    help="Filename glob under motion-dir (skips *profile*)",
  )
  args = parser.parse_args()

  motion_dir = args.motion_dir.resolve()
  out_dir = (args.out_dir or (motion_dir / "video")).resolve()
  paths = sorted(motion_dir.glob(args.glob))
  paths = [p for p in paths if "profile" not in p.stem]
  if not paths:
    raise SystemExit(f"No motions matching {args.glob!r} in {motion_dir}")

  model = _build_model(args.xml.resolve(), args.width, args.height)
  data = mujoco.MjData(model)
  joint_map = _joint_qpos_map(model)

  print(f"Rendering {len(paths)} clips → {out_dir}")
  for path in paths:
    render_npz(
      model,
      data,
      joint_map,
      path,
      out_dir / f"{path.stem}.mp4",
      width=args.width,
      height=args.height,
      camera_distance=args.distance,
    )
  print("done")


if __name__ == "__main__":
  main()
