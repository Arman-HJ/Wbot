#!/usr/bin/env python3
"""Preview Adam SP upper-body motion packs (FALCON ``upper14``) as MP4.

Legs/waist stay at HOME_KEYFRAME; only the 14 arm joints follow the clip.

Usage::

  python scripts/play_upper_body_motions.py
  python scripts/play_upper_body_motions.py --clip 0 --viewer
  python scripts/play_upper_body_motions.py --all --out logs/upper_motion_previews
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_MOTION_DIR = _REPO / "src/assets/upper_motions/pnd_adam_sp"
_DEFAULT_OUT = _REPO / "logs/upper_motion_previews"

UPPER_JOINT_NAMES = [
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
]

HOME_JOINT_POS = {
  ".*_hip_pitch_joint": -0.20,
  "left_hip_roll_joint": 0.00,
  "right_hip_roll_joint": 0.00,
  "left_hip_yaw_joint": -0.08,
  "right_hip_yaw_joint": 0.08,
  ".*_knee_joint": 0.40,
  ".*_ankle_pitch_joint": -0.20,
  "left_ankle_roll_joint": 0.00,
  "right_ankle_roll_joint": 0.00,
  "waist_roll_joint": 0.0,
  "waist_pitch_joint": 0.0,
  "waist_yaw_joint": 0.0,
  ".*_shoulder_pitch_joint": 0.0,
  ".*_elbow_joint": -0.3,
  "left_shoulder_roll_joint": 0.1,
  "right_shoulder_roll_joint": -0.1,
  ".*_wrist_.*_joint": 0.0,
}
ROOT_POS = (0.0, 0.0, 0.95)
ROOT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _build_model(xml_path: Path) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 960
  world = spec.worldbody
  world.add_light(pos=(2.0, -2.0, 3.0), dir=(-0.5, 0.5, -1.0))
  world.add_light(pos=(-2.0, 2.0, 3.0), dir=(0.5, -0.5, -1.0))
  floor = world.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = (2.0, 2.0, 0.01)
  floor.rgba = (0.35, 0.35, 0.38, 1.0)
  return spec.compile()


def _hinge_joint_names(model: mujoco.MjModel) -> list[str]:
  names: list[str] = []
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name:
      names.append(name.split("/")[-1])
  return names


def _expand_joint_pos(joint_names: list[str], patterns: dict[str, float]) -> dict[str, float]:
  out = {n: 0.0 for n in joint_names}
  for pattern, value in patterns.items():
    rx = re.compile(pattern)
    for name in joint_names:
      if rx.fullmatch(name):
        out[name] = float(value)
  return out


def _joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
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


def _apply_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_map: dict[str, int],
  base_joint_pos: dict[str, float],
  upper14: np.ndarray,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  if model.nq >= 7:
    data.qpos[0:3] = ROOT_POS
    data.qpos[3:7] = ROOT_QUAT_WXYZ
  for name, q in base_joint_pos.items():
    if name in joint_map:
      data.qpos[joint_map[name]] = q
  for i, name in enumerate(UPPER_JOINT_NAMES):
    if name in joint_map:
      data.qpos[joint_map[name]] = float(upper14[i])
  mujoco.mj_forward(model, data)


def _load_playlist(motion_dir: Path) -> list[dict]:
  with open(motion_dir / "playlist.yaml", encoding="utf-8") as handle:
    pl = yaml.safe_load(handle) or {}
  return list(pl.get("clips") or [])


def _setup_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = (0.0, 0.0, 1.05)
  cam.distance = 2.8
  cam.azimuth = 135.0
  cam.elevation = -12.0
  return cam


def render_clip_mp4(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_map: dict[str, int],
  base_joint_pos: dict[str, float],
  upper: np.ndarray,
  fps: float,
  out_path: Path,
  *,
  width: int = 960,
  height: int = 720,
  title: str = "",
) -> None:
  renderer = mujoco.Renderer(model, height=height, width=width)
  cam = _setup_camera(model, data)
  scene_option = mujoco.MjvOption()
  dt = 1.0 / max(float(fps), 1e-6)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  frames: list[np.ndarray] = []
  for t in range(int(upper.shape[0])):
    _apply_pose(model, data, joint_map, base_joint_pos, upper[t])
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    frame = renderer.render()
    if title:
      # Simple top banner via numpy (no cv2 dependency).
      banner_h = 28
      banner = np.zeros((banner_h, frame.shape[1], 3), dtype=np.uint8)
      banner[:] = (24, 24, 28)
      frame = np.vstack([banner, frame])
    frames.append(frame)
  imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
  renderer.close()
  print(f"[video] {out_path}  frames={len(frames)} fps={fps:.1f} dur={len(frames)/fps:.1f}s")


def play_viewer(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_map: dict[str, int],
  base_joint_pos: dict[str, float],
  upper: np.ndarray,
  fps: float,
  *,
  loop: bool = True,
) -> None:
  import mujoco.viewer

  dt = 1.0 / max(float(fps), 1e-6)
  frame = 0
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      _apply_pose(model, data, joint_map, base_joint_pos, upper[frame])
      viewer.sync()
      frame += 1
      if frame >= upper.shape[0]:
        frame = 0 if loop else upper.shape[0] - 1
      time.sleep(dt)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--motion-dir", type=Path, default=_DEFAULT_MOTION_DIR)
  parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
  parser.add_argument("--clip", type=int, default=-1, help="Clip index (default: all)")
  parser.add_argument("--all", action="store_true", help="Render all clips to MP4")
  parser.add_argument("--viewer", action="store_true", help="Interactive MuJoCo viewer")
  parser.add_argument("--xml", type=Path, default=_REPO / "src/assets/robots/adam_sp/xml/adam_sp.xml")
  args = parser.parse_args()

  motion_dir = args.motion_dir.expanduser().resolve()
  clips = _load_playlist(motion_dir)
  if not clips:
    raise SystemExit(f"No clips in {motion_dir / 'playlist.yaml'}")

  model = _build_model(args.xml)
  data = mujoco.MjData(model)
  joint_map = _joint_qpos_map(model)
  base_joint_pos = _expand_joint_pos(_hinge_joint_names(model), HOME_JOINT_POS)

  indices = range(len(clips)) if args.all or args.clip < 0 else [args.clip]
  if args.viewer:
    idx = 0 if args.clip < 0 else args.clip
    clip = clips[idx]
    upper = np.load(motion_dir / clip["upper14"]).astype(np.float64)
    fps = float(clip.get("fps", 30.0))
    print(f"Viewer clip [{idx}] {clip.get('slug')}  T={upper.shape[0]} fps={fps}")
    play_viewer(model, data, joint_map, base_joint_pos, upper, fps)
    return

  if not args.all and args.clip < 0:
    args.all = True

  montage_entries: list[tuple[str, Path]] = []
  for idx in indices:
    clip = clips[idx]
    slug = str(clip.get("slug") or Path(clip["upper14"]).stem)
    upper = np.load(motion_dir / clip["upper14"]).astype(np.float64)
    fps = float(clip.get("fps", 30.0))
    title = slug.replace("_", " ")
    out_path = args.out / f"{idx:02d}_{slug}.mp4"
    render_clip_mp4(
      model,
      data,
      joint_map,
      base_joint_pos,
      upper,
      fps,
      out_path,
      title=title,
    )
    montage_entries.append((title, out_path))

  # Combined preview: first ~6s of each clip.
  preview_frames: list[np.ndarray] = []
  preview_fps = 30.0
  max_frames = int(6.0 * preview_fps)
  for title, path in montage_entries:
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", preview_fps))
    step = max(int(round(src_fps / preview_fps)), 1)
    taken = 0
    for i, frame in enumerate(reader):
      if i % step != 0:
        continue
      preview_frames.append(frame)
      taken += 1
      if taken >= max_frames:
        break
    reader.close()
  preview_path = args.out / "00_all_clips_preview.mp4"
  imageio.mimsave(preview_path, preview_frames, fps=preview_fps, codec="libx264", quality=8)
  print(f"[video] {preview_path}  clips={len(montage_entries)}")


if __name__ == "__main__":
  main()
