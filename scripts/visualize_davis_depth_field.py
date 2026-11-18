#!/usr/bin/env python3
"""Capture head-camera depth on the football_jax pitch (DAVIS play env, 1 env).

Writes ``results/davis_field_depth_views.png``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.lab_api.math import quat_from_euler_xyz  # noqa: E402

from src.tasks.davis.config.wbot_20dof.env_cfg import (  # noqa: E402
  BALL_RADIUS,
  wbot_20dof_davis_shoot_env_cfg,
)
from src.tasks.davis.mdp.depth import DepthPreprocessCfg, align_depth  # noqa: E402
from src.tasks.davis.mdp import events as davis_events  # noqa: E402

OUT = ROOT / "results" / "davis_field_depth_views.png"
DEVICE = os.environ.get("DAVIS_DEPTH_DEVICE", "cuda:0")


def _set_pose(env, robot_xy, ball_xy, yaw_to_goal: bool = True) -> None:
  """Place robot/ball in env-0 local frame; face goal (−x) or face ball."""
  ids = torch.tensor([0], device=env.device)
  origins = env.scene.env_origins[ids]
  robot = env.scene["robot"]
  ball = env.scene["ball"]

  r = origins.clone()
  r[:, 0] += float(robot_xy[0])
  r[:, 1] += float(robot_xy[1])
  r[:, 2] = 0.59
  b = origins.clone()
  b[:, 0] += float(ball_xy[0])
  b[:, 1] += float(ball_xy[1])
  b[:, 2] = BALL_RADIUS

  if yaw_to_goal:
    yaw = torch.full((1,), math.pi, device=env.device)
  else:
    to_ball = b - r
    yaw = torch.atan2(to_ball[:, 1], to_ball[:, 0])
  zeros = torch.zeros_like(yaw)
  rq = quat_from_euler_xyz(zeros, zeros, yaw)
  bq = torch.zeros(1, 4, device=env.device)
  bq[:, 0] = 1.0

  davis_events._set_root_pose_vel(robot, ids, r, rq)
  davis_events._set_root_pose_vel(ball, ids, b, bq)
  davis_events.reset_goal(env, ids)
  davis_events.pin_field(env, ids)
  jp = robot.data.default_joint_pos[ids].clone()
  # Slight neck pitch so head looks toward ball/goal (not sky).
  jnames = list(robot.joint_names)
  for name, val in (("neck_yaw_joint", 0.0), ("neck_pitch_joint", 0.35)):
    if name in jnames:
      jp[:, jnames.index(name)] = val
  robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)


def _capture(env) -> tuple[np.ndarray, np.ndarray]:
  """One physics+sensor tick → raw metric depth (H,W) + actor depth (h,w)."""
  # Zero action step to refresh sensors.
  act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  env.step(act)
  sensor = env.scene["zed2i_depth"]
  raw = sensor.data.depth[0, :, :, 0].detach().float().cpu().numpy()
  depth_m = sensor.data.depth.permute(0, 3, 1, 2).float()
  cfg = getattr(env, "_davis_depth_cfg", DepthPreprocessCfg())
  actor = align_depth(depth_m, cfg, training=False)[0, 0].detach().cpu().numpy()
  return raw, actor


def _depth_to_rgb(depth: np.ndarray, *, metric: bool) -> np.ndarray:
  """False-color depth for display."""
  d = depth.astype(np.float32).copy()
  if metric:
    valid = np.isfinite(d) & (d > 0.05) & (d < 8.0)
    vis = np.zeros_like(d)
    vis[valid] = np.clip(d[valid], 0.3, 5.0)
    # Near = bright yellow, far = dark blue (common depth viz).
    t = np.zeros_like(d)
    t[valid] = (vis[valid] - 0.3) / (5.0 - 0.3)
  else:
    # Actor: [-1,1] valid, 0 = invalid/far-fill.
    valid = np.isfinite(d) & (np.abs(d) > 1e-4)
    t = np.zeros_like(d)
    t[valid] = (d[valid] + 1.0) * 0.5  # map [-1,1] → [0,1]
  # Turbo-ish via matplotlib.
  cmap = plt.get_cmap("turbo")
  rgba = cmap(np.clip(t, 0, 1))
  rgba[~valid] = (0.05, 0.05, 0.05, 1.0)
  return (rgba[..., :3] * 255).astype(np.uint8)


def main() -> None:
  OUT.parent.mkdir(parents=True, exist_ok=True)
  cfg = wbot_20dof_davis_shoot_env_cfg(play=True)
  # Also grab RGB from the same camera for side-by-side context.
  for s in cfg.scene.sensors:
    if getattr(s, "name", None) == "zed2i_depth":
      s.data_types = ("depth", "rgb")
      break

  print(f"[depth-viz] building 1-env on {DEVICE} …")
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)
  env.reset()

  views = [
    ("penalty · face ball+goal", (3.2, 0.0), (1.5, 0.0), False),
    ("penalty · angled", (3.5, 1.2), (1.5, 0.2), False),
    ("free-kick · face goal", (5.0, 0.0), (2.5, 0.8), True),
    ("midfield look down pitch", (7.0, 0.0), (4.0, 0.0), True),
  ]

  fig, axes = plt.subplots(len(views), 3, figsize=(11.5, 3.1 * len(views)))
  for i, (title, rxy, bxy, yaw_goal) in enumerate(views):
    _set_pose(env, rxy, bxy, yaw_to_goal=yaw_goal)
    # A couple of settle steps so contacts / camera catch up.
    for _ in range(3):
      raw, actor = _capture(env)

    sensor = env.scene["zed2i_depth"]
    rgb = None
    if sensor.data.rgb is not None:
      rgb = sensor.data.rgb[0].detach().cpu().numpy()
      if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    ax0, ax1, ax2 = axes[i]
    if rgb is not None:
      ax0.imshow(rgb)
      ax0.set_title(f"{title}\nRGB (zed2i)")
    else:
      ax0.text(0.5, 0.5, "no rgb", ha="center", va="center")
      ax0.set_title(title)
    ax0.axis("off")

    ax1.imshow(_depth_to_rgb(raw, metric=True))
    ax1.set_title(f"raw depth {raw.shape[1]}×{raw.shape[0]} (m→turbo)")
    ax1.axis("off")

    ax2.imshow(_depth_to_rgb(actor, metric=False))
    ax2.set_title(f"actor depth {actor.shape[1]}×{actor.shape[0]} (norm)")
    ax2.axis("off")

    # Metric stats for the ball/goal region.
    finite = raw[np.isfinite(raw) & (raw > 0)]
    print(
      f"  {title}: raw range "
      f"[{finite.min():.2f}, {finite.max():.2f}] m  mean={finite.mean():.2f}"
      if finite.size
      else f"  {title}: empty depth"
    )

  fig.suptitle(
    "DAVIS head depth on football_jax pitch (goal@x=0, field → +x)",
    fontsize=13,
  )
  fig.tight_layout()
  fig.savefig(OUT, dpi=160)
  print(f"wrote {OUT}")
  env.close()


if __name__ == "__main__":
  main()
