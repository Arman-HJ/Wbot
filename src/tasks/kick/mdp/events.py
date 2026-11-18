"""Kick-task reset helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _init_kick_ball_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  pose_xyz: torch.Tensor,
  yaw_q: torch.Tensor,
) -> None:
  device = env.device
  n = len(env_ids)
  if not hasattr(env, "_kick_ball_start_pos"):
    env._kick_ball_start_pos = torch.zeros(env.num_envs, 3, device=device)
  if not hasattr(env, "_kick_forward_axis"):
    env._kick_forward_axis = torch.zeros(env.num_envs, 3, device=device)
  if not hasattr(env, "_kick_success"):
    env._kick_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
  if not hasattr(env, "_kick_last_forward"):
    env._kick_last_forward = torch.zeros(env.num_envs, device=device)

  env._kick_ball_start_pos[env_ids] = pose_xyz
  forward_w = quat_apply(
    yaw_q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, -1)
  )
  env._kick_forward_axis[env_ids] = forward_w
  env._kick_success[env_ids] = False
  env._kick_last_forward[env_ids] = 0.0


def place_ball_at_world(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ball_name: str = "ball",
  robot_name: str = "robot",
  pos: tuple[float, float, float] = (0.095, 0.621, 0.110),
  pos_noise: tuple[float, float, float] = (0.02, 0.02, 0.01),
) -> None:
  """Place ball near a fixed per-env local position with small uniform noise (m).

  ``pos`` is relative to each env origin (same frame as the motion clip), not a
  single shared world coordinate for all parallel envs.
  """
  robot = env.scene[robot_name]
  ball = env.scene[ball_name]
  device = env.device

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=device, dtype=torch.long)

  n = len(env_ids)
  if n == 0:
    return

  center = torch.tensor(pos, device=device, dtype=torch.float32)
  noise = torch.tensor(pos_noise, device=device, dtype=torch.float32)
  # ``pos`` is in the per-env local frame (same as motion clip); offset by env origins
  # so each parallel env gets its own ball instead of stacking all at one world point.
  sample = (
    center.unsqueeze(0)
    + (2.0 * torch.rand(n, 3, device=device) - 1.0) * noise
    + env.scene.env_origins[env_ids]
  )

  pose = torch.zeros(n, 7, device=device)
  pose[:, 0:3] = sample
  pose[:, 3] = 1.0
  vel = torch.zeros(n, 6, device=device)
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

  yaw_q = yaw_quat(robot.data.root_link_quat_w[env_ids])
  _init_kick_ball_tracking(env, env_ids, pose[:, 0:3], yaw_q)


def place_ball_in_front_of_right_foot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_name: str = "ball",
  foot_body_name: str = "right_ankle_roll_link",
  forward_range: tuple[float, float] = (0.32, 0.40),
  lateral_range: tuple[float, float] = (-0.10, -0.05),
  height: float = 0.11,
  forward_offsets: torch.Tensor | None = None,
  lateral_offsets: torch.Tensor | None = None,
) -> None:
  """Place the ball at the right-front of the right foot (call AFTER robot pose).

  Body frame: +x forward, +y left. Default lateral is negative (to the right)
  so the ball sits in 右前方 of the kicking foot.
  """
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_name]
  device = env.device

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=device, dtype=torch.long)

  n = len(env_ids)
  if n == 0:
    return

  root_quat = robot.data.root_link_quat_w[env_ids]
  yaw_q = yaw_quat(root_quat)

  body_ids, _ = robot.find_bodies(foot_body_name)
  foot_id = int(body_ids[0])
  foot_pos = robot.data.body_link_pos_w[env_ids, foot_id]

  if forward_offsets is not None and lateral_offsets is not None:
    fwd = forward_offsets.to(device=device, dtype=torch.float32).reshape(n)
    lat = lateral_offsets.to(device=device, dtype=torch.float32).reshape(n)
  else:
    fwd = torch.empty(n, device=device).uniform_(*forward_range)
    lat = torch.empty(n, device=device).uniform_(*lateral_range)
  offset_b = torch.stack((fwd, lat, torch.zeros(n, device=device)), dim=-1)
  offset_w = quat_apply(yaw_q, offset_b)

  pose = torch.zeros(n, 7, device=device)
  pose[:, 0:3] = foot_pos + offset_w
  pose[:, 2] = height
  pose[:, 3] = 1.0  # identity quat (wxyz)
  vel = torch.zeros(n, 6, device=device)
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

  _init_kick_ball_tracking(env, env_ids, pose[:, 0:3], yaw_q)


# Back-compat alias (old event name); prefer place_ball_in_front_of_right_foot.
reset_ball_in_front = place_ball_in_front_of_right_foot
