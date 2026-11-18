#!/usr/bin/env python3
"""Convert football CSV motions → WBOT 12DoF npz + preview mp4."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from mjlab.entity import Entity
from src.assets.robots import get_wbot_12dof_robot_cfg

FPS = 50
FOOTBALL_DIR = Path("/home/songhat/code_workspace/football/data")
OUT_DIR = Path(
  "/home/songhat/code_workspace/unitree_rl_mjlab/src/assets/motions/"
  "wbot_12dof/football_csv"
)
XML_SCENE = Path(
  "/home/songhat/code_workspace/unitree_rl_mjlab/src/assets/robots/wbot/xmls/scene.xml"
)

WBOT_JOINTS = [
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
]
BODY_NAMES = [
  "pelvis",
  "torso_link",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
]


def load_csv(path: Path) -> np.ndarray:
  with open(path) as f:
    rows = list(csv.reader(f))
  return np.array(rows[1:], dtype=np.float64)


def convert_one(
  path: Path,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  jnt_qposadr: dict[str, int],
  jnt_dofadr: dict[str, int],
  body_ids: list[int],
  foot_geom_ids: list[int],
) -> tuple[dict[str, np.ndarray], float]:
  arr = load_csv(path)
  root_pos = arr[:, 0:3].copy()
  root_quat = arr[:, [6, 3, 4, 5]].copy()  # xyzw → wxyz
  root_lin = arr[:, 7:10].copy()
  root_ang = arr[:, 10:13].copy()
  dof = arr[:, 13:26]  # waist + 12
  ddof = arr[:, 26:39]
  jp = dof[:, 1:13].astype(np.float32)  # drop waist
  jv = ddof[:, 1:13].astype(np.float32)

  # Start at origin in XY.
  root_pos[:, 0:2] -= root_pos[0, 0:2]

  T = jp.shape[0]
  bp = np.zeros((T, len(BODY_NAMES), 3), np.float32)
  bq = np.zeros((T, len(BODY_NAMES), 4), np.float32)
  blv = np.zeros_like(bp)
  bav = np.zeros_like(bp)

  # First pass FK to measure foot penetration / float.
  foot_z_min = np.full(T, np.inf, dtype=np.float64)
  for t in range(T):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat[t]
    for jn, val in zip(WBOT_JOINTS, jp[t]):
      data.qpos[jnt_qposadr[jn]] = val
    mujoco.mj_forward(model, data)
    for gid in foot_geom_ids:
      # geom AABB approx via xpos of geom; use geom_xpos + size for sphere/capsule
      # simpler: body ankle_roll z of geom attachments via geom_xpos[:,2] - radius
      z = float(data.geom_xpos[gid, 2])
      # capsule radius stored in geom_size[0]
      r = float(model.geom_size[gid, 0])
      foot_z_min[t] = min(foot_z_min[t], z - r)

  # Constant ground offset: lower so the lowest foot contact sits on z=0.
  # (matches prior football→wbot kick grounding approach)
  z_off = float(np.min(foot_z_min))
  root_pos[:, 2] -= z_off

  for t in range(T):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat[t]
    for jn, val in zip(WBOT_JOINTS, jp[t]):
      data.qpos[jnt_qposadr[jn]] = val
    for jn, val in zip(WBOT_JOINTS, jv[t]):
      data.qvel[jnt_dofadr[jn]] = val
    data.qvel[0:3] = root_lin[t]
    data.qvel[3:6] = root_ang[t]
    mujoco.mj_forward(model, data)
    for bi, bid in enumerate(body_ids):
      bp[t, bi] = data.xpos[bid]
      bq[t, bi] = data.xquat[bid]
      vel = np.zeros(6)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
      bav[t, bi] = vel[0:3]
      blv[t, bi] = vel[3:6]

  payload = {
    "fps": np.array([FPS], dtype=np.int64),
    "joint_names": np.array(WBOT_JOINTS),
    "body_names": np.array(BODY_NAMES),
    "joint_pos": jp,
    "joint_vel": jv,
    "body_pos_w": bp,
    "body_quat_w": bq,
    "body_lin_vel_w": blv,
    "body_ang_vel_w": bav,
  }
  return payload, z_off


def render_mp4(
  payload: dict[str, np.ndarray],
  out_mp4: Path,
  scene_xml: Path,
) -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  model = mujoco.MjModel.from_xml_path(str(scene_xml))
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=480, width=640)

  jp = payload["joint_pos"]
  bp = payload["body_pos_w"]
  bq = payload["body_quat_w"]
  joint_names = [str(x) for x in payload["joint_names"].tolist()]
  body_names = [str(x) for x in payload["body_names"].tolist()]
  pelvis_i = body_names.index("pelvis")

  jnt_qposadr = {}
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and "floating" not in name:
      jnt_qposadr[name] = int(model.jnt_qposadr[i])

  # freejoint may be named differently in scene
  free_qadr = 0
  for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      free_qadr = int(model.jnt_qposadr[i])
      break

  frames = []
  T = jp.shape[0]
  for t in range(T):
    data.qpos[:] = 0.0
    data.qpos[free_qadr : free_qadr + 3] = bp[t, pelvis_i]
    data.qpos[free_qadr + 3 : free_qadr + 7] = bq[t, pelvis_i]
    for jn, val in zip(joint_names, jp[t]):
      if jn in jnt_qposadr:
        data.qpos[jnt_qposadr[jn]] = val
    mujoco.mj_forward(model, data)

    # Look-at pelvis
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = bp[t, pelvis_i]
    cam.distance = 2.5
    cam.azimuth = 135
    cam.elevation = -15
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

  renderer.close()
  imageio.mimsave(out_mp4, frames, fps=FPS)
  print(f"  video {out_mp4.name} ({T} frames)")


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  robot = Entity(get_wbot_12dof_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)

  jnt_qposadr: dict[str, int] = {}
  jnt_dofadr: dict[str, int] = {}
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and name != "floating_base_joint":
      jnt_qposadr[name] = int(model.jnt_qposadr[i])
      jnt_dofadr[name] = int(model.jnt_dofadr[i])

  body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODY_NAMES
  ]
  assert all(i >= 0 for i in body_ids)

  foot_geom_ids = []
  for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if "foot" in name and "collision" in name:
      foot_geom_ids.append(i)
  assert foot_geom_ids, "no foot collision geoms"

  csvs = sorted(FOOTBALL_DIR.glob("*.csv"))
  print(f"converting {len(csvs)} clips → {OUT_DIR}")
  for path in csvs:
    stem = path.stem
    payload, z_off = convert_one(
      path, model, data, jnt_qposadr, jnt_dofadr, body_ids, foot_geom_ids
    )
    out_npz = OUT_DIR / f"{stem}_wbot_12dof.npz"
    np.savez_compressed(out_npz, **payload)
    pel_z = payload["body_pos_w"][:, 0, 2]
    print(
      f"{stem}: T={payload['joint_pos'].shape[0]} "
      f"z_off={z_off:.3f} pelvis_z=[{pel_z.min():.3f},{pel_z.max():.3f}] → {out_npz.name}"
    )
    out_mp4 = OUT_DIR / f"{stem}_wbot_12dof.mp4"
    render_mp4(payload, out_mp4, XML_SCENE)

  print("done")


if __name__ == "__main__":
  main()
