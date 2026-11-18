"""Kick-task rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand

from src.tasks.amp.mdp import rewards as amp_rewards

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_foot_pos_z_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.1,
  body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Exp kernel on foot height error vs motion (relative Z only)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idxs = [
    i
    for i, name in enumerate(command.cfg.body_names)
    if name in body_names
  ]
  if not idxs:
    raise ValueError(f"foot bodies {body_names} not in motion body_names")
  err_z = (
    command.body_pos_relative_w[:, idxs, 2]
    - command.robot_body_pos_w[:, idxs, 2]
  )
  # (N, B) → mean over feet, same form as body pos exp rewards.
  return torch.exp(-torch.square(err_z).mean(-1) / (std**2))


def _ball_travel_distance(env: ManagerBasedRlEnv, ball_name: str) -> torch.Tensor:
  """Horizontal distance (XY) of the ball from its reset position."""
  ball = env.scene[ball_name]
  start = env._kick_ball_start_pos
  delta_xy = ball.data.root_link_pos_w[:, :2] - start[:, :2]
  return torch.linalg.norm(delta_xy, dim=-1)


def ball_forward_progress(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  success_distance: float = 3.0,
) -> torch.Tensor:
  """Dense reward for ball travel distance (clipped by success distance)."""
  dist = _ball_travel_distance(env, ball_name).clamp(min=0.0, max=success_distance)
  if not hasattr(env, "_kick_last_forward"):
    env._kick_last_forward = dist.clone()
  progress = (dist - env._kick_last_forward).clamp(min=0.0)
  env._kick_last_forward = dist
  return progress


def kick_success_bonus(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  success_distance: float = 3.0,
) -> torch.Tensor:
  """Sparse bonus the first time ball travel distance exceeds ``success_distance``."""
  dist = _ball_travel_distance(env, ball_name)
  newly = (dist > success_distance) & (~env._kick_success)
  env._kick_success |= newly
  return newly.float()


def no_kick_timeout_penalty(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  min_forward: float = 0.3,
) -> torch.Tensor:
  """Penalty when the episode times out without a meaningful kick."""
  timed_out = env.termination_manager.time_outs
  dist = _ball_travel_distance(env, ball_name)
  bad = timed_out & (dist < min_forward) & (~env._kick_success)
  return bad.float()


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize ankle xy speed while the foot is in contact (anti-slip)."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  contact = force > force_threshold
  speed = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
  )
  return torch.sum(speed * contact.float(), dim=1)


def _stand_still_weight(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  torso_cfg: SceneEntityCfg,
  *,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  """AMP-style stand gate without twist command (kick uses motion command)."""
  asset: Entity = env.scene[torso_cfg.name]
  planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  quat = asset.data.body_link_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_w = asset.data.body_link_ang_vel_w[:, torso_cfg.body_ids, :].squeeze(1)
  torso_ang_speed = torch.linalg.vector_norm(
    quat_apply_inverse(quat, ang_vel_w)[:, :2], dim=1
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  peak_force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  double_support = torch.all(peak_force > contact_force_threshold, dim=1)
  motion_weight = torch.exp(-torch.square(planar_speed / planar_speed_scale))
  motion_weight = motion_weight * torch.exp(
    -torch.square(torso_ang_speed / torso_ang_vel_scale)
  )
  return double_support.float() * motion_weight


def stand_torso_pitch_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Same as AMP stand_torso_pitch, gated by kick stand-still weight."""
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.square(gravity_b[:, 0]) * _stand_still_weight(
    env, sensor_name, asset_cfg
  )


def stand_default_joint_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Hold default joint pose while standing (AMP stand_hip_yaw / shoulder_*)."""
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(error), dim=1) * _stand_still_weight(
    env, sensor_name, torso_cfg
  )


def stand_feet_heading_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Keep feet heading aligned with pelvis while standing (AMP stand_feet_heading)."""
  asset: Entity = env.scene[feet_cfg.name]
  feet_quat_w = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
  foot_forward_b = torch.zeros(
    (*feet_quat_w.shape[:-1], 3), device=env.device, dtype=feet_quat_w.dtype
  )
  foot_forward_b[..., 0] = 1.0
  foot_forward_w = quat_apply(feet_quat_w, foot_forward_b)
  pelvis_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand_as(feet_quat_w)
  foot_forward_pelvis = quat_apply_inverse(pelvis_quat_w, foot_forward_w)
  heading_error = torch.atan2(
    foot_forward_pelvis[..., 1], foot_forward_pelvis[..., 0]
  )
  return torch.sum(torch.square(heading_error), dim=1) * _stand_still_weight(
    env, sensor_name, torso_cfg
  )


def stable_stand_hip_roll_norm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.linalg.vector_norm(error, dim=1) * _stand_still_weight(
    env, sensor_name, torso_cfg
  )


def stable_stand_feet_fore_aft_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
  position_deadband: float = 0.02,
) -> torch.Tensor:
  feet = amp_rewards._feet_positions_b(env.scene[feet_cfg.name], feet_cfg)
  error = torch.sum(torch.relu(torch.abs(feet[:, :, 0]) - position_deadband), dim=1)
  return error * _stand_still_weight(env, sensor_name, torso_cfg)


def stable_stand_feet_lateral_center_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
  center_deadband: float = 0.01,
) -> torch.Tensor:
  feet = amp_rewards._feet_positions_b(env.scene[feet_cfg.name], feet_cfg)
  center = 0.5 * (feet[:, 0, 1] + feet[:, 1, 1])
  error = torch.relu(torch.abs(center) - center_deadband)
  return error * _stand_still_weight(env, sensor_name, torso_cfg)
