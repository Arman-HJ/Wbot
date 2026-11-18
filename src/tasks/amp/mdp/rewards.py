"""Task rewards accompanying the adversarial walking-style reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .commands import (
  COMMAND_MODE_LATERAL,
  COMMAND_MODE_MIXED,
  COMMAND_MODE_TURNING,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  command = env.command_manager.get_command(name)
  assert command is not None
  return command


def moving_mask(
  env: ManagerBasedRlEnv, command_name: str = "twist", threshold: float = 0.05
) -> torch.Tensor:
  command = _command(env, command_name)
  return (
    torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  ) > threshold


def _walk_style_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  minimum_command: float,
  maximum_forward_speed: float,
  include_turning: bool,
) -> torch.Tensor:
  """Select moving walk-style commands without constraining the run gait."""
  command = _command(env, command_name)
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError(
      "Walk-style foot rewards require UniformVelocityCommand.command_mode."
    )
  command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
    command[:, 2]
  )
  walk = (command_mode == COMMAND_MODE_MIXED) & (
    command[:, 0] <= maximum_forward_speed
  )
  if include_turning:
    walk = walk | (command_mode == COMMAND_MODE_TURNING)
  return walk & (command_magnitude > minimum_command)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_command: float = 0.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
  reward = torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)
  if min_command > 0.0:
    lin = torch.linalg.vector_norm(command[:, :2], dim=1)
    reward = reward * (lin >= min_command).float()
  return reward


def track_yaw_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_command: float = 0.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
  reward = torch.exp(-torch.square(error) / std**2)
  if min_command > 0.0:
    reward = reward * (torch.abs(command[:, 2]) >= min_command).float()
  return reward


def yaw_rate_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_command: float = 0.0,
) -> torch.Tensor:
  """Squared yaw-rate error once a yaw command is present.

  The exponential tracking term is nearly flat when the robot ignores a
  turn, so it never starts rotating. This cost still has a slope there.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
  cost = torch.square(error)
  if min_command > 0.0:
    cost = cost * (torch.abs(command[:, 2]) >= min_command).float()
  return cost


def lin_vel_xy_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_command: float = 0.0,
) -> torch.Tensor:
  """Squared planar velocity error (same role as ``yaw_rate_error_l2``).

  Exp tracking stays high when the robot ignores a small forward command;
  this L2 term keeps a slope so low-speed starts are not preferred.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
  cost = torch.sum(torch.square(error), dim=1)
  if min_command > 0.0:
    lin = torch.linalg.vector_norm(command[:, :2], dim=1)
    cost = cost * (lin >= min_command).float()
  return cost


def _slow_straight_command_term(env: ManagerBasedRlEnv, command_name: str):
  term = env.command_manager.get_term(command_name)
  if not hasattr(term, "is_slow_walk_env") or not hasattr(
    term, "slow_walk_path_offset"
  ):
    raise RuntimeError(
      "straight_walk rewards require UniformVelocityCommand slow-walk sampling "
      "(rel_slow_walk_envs + path origin)."
    )
  return term


def straight_walk_reward(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_forward_progress: float = 0.05,
) -> torch.Tensor:
  """Reward staying near the slow-walk start line (lateral path offset).

  Only ``is_slow_walk_env`` samples. Uses world-frame lateral distance to the
  heading line frozen at sample time — not body ``vy``/``wz``, which cannot be
  zero in a real gait. Progress uses along-track displacement.
  """
  del asset_cfg  # path uses command-term robot; kept for RewardTermCfg uniformity
  term = _slow_straight_command_term(env, command_name)
  command = _command(env, command_name)
  lateral, forward = term.slow_walk_path_offset()
  straight = torch.exp(-torch.square(lateral) / std**2)
  # Normalize progress by commanded distance scale (~1s of cmd); clamp to [0,1].
  progress = torch.clamp(
    forward / command[:, 0].clamp(min=1.0e-3), min=0.0, max=1.0
  )
  gate = term.is_slow_walk_env & (forward >= min_forward_progress)
  return straight * progress * gate.float()


def straight_walk_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_forward_progress: float = 0.05,
  deadband: float = 0.01,
) -> torch.Tensor:
  """Squared lateral path offset beyond a deadband (slow-walk samples).

  Weight is curriculum-ramped (see ``straight_walk_penalty_stages``); keep the
  raw term as unbounded ``excess²`` so late training keeps a strong slope.
  """
  del asset_cfg
  term = _slow_straight_command_term(env, command_name)
  lateral, forward = term.slow_walk_path_offset()
  excess = torch.relu(torch.abs(lateral) - deadband)
  gate = term.is_slow_walk_env & (forward >= min_forward_progress)
  return torch.square(excess) * gate.float()


def heading_hold(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward holding the slow-walk start heading (bounded exp of yaw error).

  Prefer this over lateral path distance: yaw bias is the root cause of
  straight-walk drift, and heading error is one integration closer to the
  actuator than path offset.
  """
  del asset_cfg
  term = _slow_straight_command_term(env, command_name)
  err = term.slow_walk_heading_error()
  return torch.exp(-torch.square(err) / std**2) * term.is_slow_walk_env.float()


def vertical_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_angular_velocity_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def energy_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  power = torch.abs(
    asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.linalg.vector_norm(power, dim=1)


def torso_orientation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.sum(torch.square(gravity_b[:, :2]), dim=1)


def torso_ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(quat_apply_inverse(quat, vel_w)[:, :2]), dim=1)


def torso_roll_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.square(gravity_b[:, 1])


def torso_roll_ang_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.square(quat_apply_inverse(quat, vel_w)[:, 0])


def stand_torso_pitch_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
  threshold: float = 0.1,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return (
    torch.square(gravity_b[:, 0])
    * (~moving_mask(env, command_name, threshold=threshold)).float()
  )


def zero_command_joint_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
  threshold: float = 0.05,
  lin_threshold: float | None = None,
  yaw_threshold: float | None = None,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  command = _command(env, command_name)
  if lin_threshold is not None and yaw_threshold is not None:
    lin = torch.linalg.vector_norm(command[:, :2], dim=1)
    yaw = torch.abs(command[:, 2])
    gate = (lin < lin_threshold) & (yaw < yaw_threshold)
  else:
    magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    gate = magnitude < threshold
  return torch.sum(torch.square(error), dim=1) * gate.float()


def _feet_positions_b(asset: Entity, feet_cfg: SceneEntityCfg) -> torch.Tensor:
  pos_w = asset.data.body_link_pos_w[:, feet_cfg.body_ids, :]
  relative = pos_w - asset.data.root_link_pos_w.unsqueeze(1)
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, len(feet_cfg.body_ids), -1)
  return quat_apply_inverse(quat, relative)


def feet_crossing(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  signed_distance = feet[:, 0, 1] - feet[:, 1, 1]
  return torch.relu(minimum_distance - signed_distance)


def feet_spacing_non_lateral(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  maximum_distance: float,
  lateral_command_threshold: float,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
  disable_turning: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  distance = feet[:, 0, 1] - feet[:, 1, 1]
  outside = torch.relu(minimum_distance - distance) + torch.relu(
    distance - maximum_distance
  )
  non_lateral = torch.abs(_command(env, command_name)[:, 1]) < lateral_command_threshold
  gate = non_lateral.float()
  if disable_turning:
    term = env.command_manager.get_term(command_name)
    command_mode = getattr(term, "command_mode", None)
    if command_mode is not None:
      gate = gate * (command_mode != COMMAND_MODE_TURNING).float()
  return outside * gate


def stand_feet_heading_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
  threshold: float = 0.05,
) -> torch.Tensor:
  """Keep both feet aligned with the pelvis while standing still."""
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
  command = _command(env, command_name)
  magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return torch.sum(torch.square(heading_error), dim=1) * (magnitude < threshold).float()


def feet_pelvis_heading_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
  turn_yaw_scale: float = 0.5,
  turn_floor: float = 0.3,
) -> torch.Tensor:
  """Penalize foot heading away from the pelvis.

  Straight and lateral commands keep the full penalty. A larger yaw-rate
  command scales it down toward ``turn_floor`` instead of turning it off.
  Pure turning (command_mode) is fully exempt so the swing foot can yaw.
  """
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
  yaw = torch.abs(_command(env, command_name)[:, 2])
  turn_blend = yaw / (yaw + turn_yaw_scale)
  scale = 1.0 - turn_blend * (1.0 - turn_floor)
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is not None:
    scale = scale * (command_mode != COMMAND_MODE_TURNING).float()
  return torch.sum(torch.square(heading_error), dim=1) * scale


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = "twist",
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at first contact (landing_force)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  threshold: float = 1.0,
  command_name: str = "twist",
  disable_turning: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  contact = force > threshold
  speed = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2], dim=-1
  )
  cost = torch.sum(speed * contact.float(), dim=1)
  if disable_turning:
    term = env.command_manager.get_term(command_name)
    command_mode = getattr(term, "command_mode", None)
    if command_mode is not None:
      cost = cost * (command_mode != COMMAND_MODE_TURNING).float()
  return cost


def walk_swing_foot_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  contact_height: float,
  target_clearance: float,
  minimum_air_time: float = 0.16,
  maximum_air_time: float = 0.36,
  minimum_support_time: float = 0.04,
  command_name: str = "twist",
  minimum_command: float = 0.1,
  maximum_forward_speed: float = 0.8,
  include_turning: bool = True,
) -> torch.Tensor:
  """Penalize supported swing feet that remain below the target clearance."""
  if target_clearance <= 0.0:
    raise ValueError("target_clearance must be positive.")
  if maximum_air_time <= minimum_air_time:
    raise ValueError("maximum_air_time must exceed minimum_air_time.")
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  assert sensor.data.current_contact_time is not None
  foot_height = asset.data.body_link_pos_w[:, feet_cfg.body_ids, 2]
  clearance = foot_height - contact_height
  clearance_deficit = torch.relu(target_clearance - clearance) / target_clearance
  clearance_deficit = torch.clamp(clearance_deficit, max=1.0)
  swing = (sensor.data.current_air_time > minimum_air_time) & (
    sensor.data.current_air_time < maximum_air_time
  )
  support = torch.flip(sensor.data.current_contact_time, dims=(1,))
  supported_swing = swing & (support > minimum_support_time)
  gait_mask = _walk_style_mask(
    env,
    command_name,
    minimum_command,
    maximum_forward_speed,
    include_turning,
  )
  return (
    torch.mean(torch.square(clearance_deficit) * supported_swing.float(), dim=1)
    * gait_mask.float()
  )


def walk_air_time_tracking(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_air_time: float,
  std: float,
  minimum_support_time: float = 0.04,
  command_name: str = "twist",
  minimum_command: float = 0.1,
  maximum_forward_speed: float = 0.8,
  include_turning: bool = True,
) -> torch.Tensor:
  """Penalize walk-style landings outside the expert swing duration."""
  if std <= 0.0:
    raise ValueError("std must be positive.")
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.last_air_time is not None
  assert sensor.data.current_contact_time is not None
  first_contact = sensor.compute_first_contact(env.step_dt)
  error = torch.abs(sensor.data.last_air_time - target_air_time) / std
  error = torch.clamp(error, max=1.0)
  support = torch.flip(sensor.data.current_contact_time, dims=(1,))
  supported_landing = first_contact & (support > minimum_support_time)
  landing_error = torch.square(error) * supported_landing.float()
  gait_mask = _walk_style_mask(
    env,
    command_name,
    minimum_command,
    maximum_forward_speed,
    include_turning,
  )
  return torch.sum(landing_error, dim=1) * gait_mask.float()


def touchdown_foot_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  velocity_deadband: float,
  maximum_velocity: float,
  command_name: str = "twist",
  minimum_command: float = 0.1,
  maximum_forward_speed: float = 0.8,
  include_turning: bool = True,
) -> torch.Tensor:
  """Penalize excessive horizontal foot speed at the first contact step."""
  if maximum_velocity <= velocity_deadband:
    raise ValueError("maximum_velocity must exceed velocity_deadband.")
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(env.step_dt)
  speed = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2], dim=-1
  )
  excess = torch.relu(speed - velocity_deadband)
  excess = torch.clamp(excess, max=maximum_velocity - velocity_deadband)
  gait_mask = _walk_style_mask(
    env,
    command_name,
    minimum_command,
    maximum_forward_speed,
    include_turning,
  )
  return (
    torch.sum(torch.square(excess) * first_contact.float(), dim=1)
    * gait_mask.float()
  )


def slow_walk_heel_first_touchdown(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
  heel_offset: tuple[float, float, float] = (-0.035, 0.0, -0.027),
  toe_offset: tuple[float, float, float] = (0.110, 0.0, -0.027),
  std: float = 0.01,
  use_slow_walk_flag: bool = False,
  min_forward_speed: float = 0.13,
  max_forward_speed: float = 0.8,
  lateral_eps: float = 0.05,
  yaw_eps: float = 0.1,
) -> torch.Tensor:
  """Reward heel-lower-than-toe attitude at first contact during non-run walk.

  Foot capsules live on ``*_ankle_roll_link``: heel ≈ x=-0.035, toe ≈ x=+0.11,
  sole z=-0.027. Positive reward when toe world-z exceeds heel world-z.
  Flat foot (dz≈0) scores 0; toe-strike does not get credit.
  """
  if std <= 0.0:
    raise ValueError("std must be positive.")
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(env.step_dt)
  feet_pos = asset.data.body_link_pos_w[:, feet_cfg.body_ids, :]
  feet_quat = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
  heel_b = torch.tensor(heel_offset, device=env.device, dtype=feet_pos.dtype)
  toe_b = torch.tensor(toe_offset, device=env.device, dtype=feet_pos.dtype)
  heel_b = heel_b.view(1, 1, 3).expand(feet_pos.shape[0], feet_pos.shape[1], -1)
  toe_b = toe_b.view(1, 1, 3).expand_as(heel_b)
  heel_z = (feet_pos + quat_apply(feet_quat, heel_b))[..., 2]
  toe_z = (feet_pos + quat_apply(feet_quat, toe_b))[..., 2]
  # Heel first ⇒ heel below toe ⇒ (toe_z - heel_z) > 0.
  # Center sigmoid at 0 and drop the negative half so flat/toe-strike ≠ free reward.
  per_foot = torch.relu(torch.sigmoid((toe_z - heel_z) / std) - 0.5) * 2.0
  reward = torch.sum(per_foot * first_contact.float(), dim=1)

  term = env.command_manager.get_term(command_name)
  is_slow = getattr(term, "is_slow_walk_env", None)
  if use_slow_walk_flag and is_slow is not None:
    gate = is_slow
  else:
    command = _command(env, command_name)
    gate = (
      (command[:, 0] >= min_forward_speed)
      & (command[:, 0] <= max_forward_speed)
      & (torch.abs(command[:, 1]) <= lateral_eps)
      & (torch.abs(command[:, 2]) <= yaw_eps)
    )
  return reward * gate.float()


def turn_lateral_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  minimum_air_time: float,
  full_reward_air_time: float,
  minimum_support_time: float = 0.04,
  command_name: str = "twist",
) -> torch.Tensor:
  """Reward alternating single-support steps during turn or lateral commands."""
  if full_reward_air_time <= minimum_air_time:
    raise ValueError("full_reward_air_time must exceed minimum_air_time.")
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.last_air_time is not None
  assert sensor.data.current_contact_time is not None
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError("Turn air-time reward requires command_mode.")
  first_contact = sensor.compute_first_contact(env.step_dt)
  support = torch.flip(sensor.data.current_contact_time, dims=(1,))
  valid_landing = first_contact & (support > minimum_support_time)
  progress = (
    sensor.data.last_air_time - minimum_air_time
  ) / (full_reward_air_time - minimum_air_time)
  progress = torch.clamp(progress, min=0.0, max=1.0)
  reward = torch.sum(progress * valid_landing.float(), dim=1)
  non_forward_mode = (command_mode == COMMAND_MODE_TURNING) | (
    command_mode == COMMAND_MODE_LATERAL
  )
  return reward * non_forward_mode.float()


def turn_clearance_lift(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
  target_clearance: float = 0.06,
  std: float = 0.03,
  min_yaw_command: float = 0.1,
  minimum_air_time: float = 0.08,
  minimum_contact_time: float = 0.02,
  command_name: str = "twist",
) -> torch.Tensor:
  """Dense reward for swing-foot clearance during pure turning.

  Requires true single support with the swing foot airborne for at least
  ``minimum_air_time``. Double-support scuff/pivot hacks get zero.
  """
  if target_clearance <= 0.0:
    raise ValueError("target_clearance must be positive.")
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  assert sensor.data.current_air_time is not None
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError("Turn clearance lift requires command_mode.")
  command = _command(env, command_name)
  turning = (command_mode == COMMAND_MODE_TURNING) & (
    torch.abs(command[:, 2]) >= min_yaw_command
  )
  contact = sensor.data.current_contact_time > minimum_contact_time
  single_support = contact.sum(dim=1) == 1
  swing_air = torch.where(
    contact,
    torch.zeros_like(sensor.data.current_air_time),
    sensor.data.current_air_time,
  )
  swinging = (swing_air >= minimum_air_time).any(dim=1)
  valid = turning & single_support & swinging

  foot_z = asset.data.body_link_pos_w[:, feet_cfg.body_ids, 2]
  support = torch.minimum(foot_z[:, 0], foot_z[:, 1])
  peak = torch.maximum(foot_z[:, 0], foot_z[:, 1]) - support
  shortfall = torch.relu(target_clearance - peak)
  reward = torch.exp(-torch.square(shortfall) / (std * std))
  return reward * valid.float()


def turn_lateral_contact_pattern(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  minimum_contact_time: float = 0.02,
  command_name: str = "twist",
) -> torch.Tensor:
  """Reward exactly one supporting foot during pure turning or lateral motion."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError("Turn contact reward requires command_mode.")
  contact = sensor.data.current_contact_time > minimum_contact_time
  single_support = contact.sum(dim=1) == 1
  non_forward_mode = (command_mode == COMMAND_MODE_TURNING) | (
    command_mode == COMMAND_MODE_LATERAL
  )
  return single_support.float() * non_forward_mode.float()


def turn_lateral_air_time_dense(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  turn_air_time_threshold: float,
  lateral_air_time_threshold: float,
  minimum_contact_time: float = 0.02,
  command_name: str = "twist",
) -> torch.Tensor:
  """Reward swing-foot air time during valid turn/lateral single support."""
  if turn_air_time_threshold <= 0.0 or lateral_air_time_threshold <= 0.0:
    raise ValueError("Air-time thresholds must be positive.")
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  assert sensor.data.current_air_time is not None
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError("Turn air-time reward requires command_mode.")
  contact = sensor.data.current_contact_time > minimum_contact_time
  single_support = contact.sum(dim=1) == 1
  swing_air_time = torch.where(
    contact,
    torch.zeros_like(sensor.data.current_air_time),
    sensor.data.current_air_time,
  )
  threshold = torch.where(
    command_mode == COMMAND_MODE_TURNING,
    torch.full_like(swing_air_time[:, 0], turn_air_time_threshold),
    torch.full_like(swing_air_time[:, 0], lateral_air_time_threshold),
  )
  within_target = swing_air_time <= threshold.unsqueeze(1)
  dense_reward = (
    (swing_air_time / threshold.unsqueeze(1)) * within_target.float()
  ).sum(dim=1)
  non_forward_mode = (command_mode == COMMAND_MODE_TURNING) | (
    command_mode == COMMAND_MODE_LATERAL
  )
  valid = single_support & non_forward_mode
  return dense_reward * valid.float()


def turn_swing_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  min_air_time: float = 0.32,
  target_clearance: float = 0.10,
  contact_height: float = 0.06,
  scale: float = 80.0,
  command_name: str = "twist",
) -> torch.Tensor:
  """Reward high swing clearance during turn/lateral — only after a long swing.

  Shorter swings get zero so scuffing cannot collect task reward.
  """
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  term = env.command_manager.get_term(command_name)
  command_mode = getattr(term, "command_mode", None)
  if command_mode is None:
    raise RuntimeError("Turn clearance reward requires command_mode.")
  non_forward = (command_mode == COMMAND_MODE_TURNING) | (
    command_mode == COMMAND_MODE_LATERAL
  )
  in_swing = sensor.data.current_air_time > 0.02
  long_swing = sensor.data.current_air_time >= min_air_time
  foot_z = asset.data.body_link_pos_w[:, feet_cfg.body_ids, 2]
  clearance = foot_z - contact_height
  shortfall = torch.clamp(target_clearance - clearance, min=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  per_foot = shaped * in_swing.float() * long_swing.float()
  return torch.sum(per_foot, dim=1) * non_forward.float()


class turn_foot_height_tracking:
  """Dense foot-clearance tracking against turn expert phase profiles.

  Loads ``turn_r`` / ``turn_l`` height curves (one gait cycle). While the
  command is pure turning, advances a per-env phase at the expert step rate
  and rewards matching left/right clearance to the profile every step.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    from pathlib import Path

    import numpy as np

    del cfg
    root = Path(__file__).resolve().parents[4]  # DroidUpE1_mjlab/
    right_path = (
      root / "src/assets/amp_motions/wbot_20dof/walk_run_amp/turn_r_foot_height_profile.npz"
    )
    left_path = (
      root / "src/assets/amp_motions/wbot_20dof/walk_run_amp/turn_l_foot_height_profile.npz"
    )
    right = np.load(right_path)
    left = np.load(left_path)
    self._right_l = torch.as_tensor(right["left"], device=env.device, dtype=torch.float32)
    self._right_r = torch.as_tensor(right["right"], device=env.device, dtype=torch.float32)
    self._left_l = torch.as_tensor(left["left"], device=env.device, dtype=torch.float32)
    self._left_r = torch.as_tensor(left["right"], device=env.device, dtype=torch.float32)
    self._period_right = float(right["period_s"])
    self._period_left = float(left["period_s"])
    self._n_bins = int(right["n_bins"])
    self._phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._step_dt = float(env.step_dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._phase.zero_()
    else:
      self._phase[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    feet_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.03,
    min_yaw_command: float = 0.1,
  ) -> torch.Tensor:
    asset: Entity = env.scene[feet_cfg.name]
    term = env.command_manager.get_term(command_name)
    command_mode = getattr(term, "command_mode", None)
    if command_mode is None:
      raise RuntimeError("Turn foot-height tracking requires command_mode.")
    command = _command(env, command_name)
    yaw = command[:, 2]
    turning = (command_mode == COMMAND_MODE_TURNING) & (
      torch.abs(yaw) >= min_yaw_command
    )
    # Advance phase only while turning; period from matching expert clip.
    period = torch.where(
      yaw >= 0.0,
      torch.full_like(yaw, self._period_left),
      torch.full_like(yaw, self._period_right),
    )
    self._phase = torch.where(
      turning,
      torch.remainder(self._phase + self._step_dt / period, 1.0),
      self._phase,
    )
    idx = torch.clamp(
      (self._phase * self._n_bins).long(), max=self._n_bins - 1
    )
    use_left_turn = yaw >= 0.0
    target_l = torch.where(use_left_turn, self._left_l[idx], self._right_l[idx])
    target_r = torch.where(use_left_turn, self._left_r[idx], self._right_r[idx])
    foot_z = asset.data.body_link_pos_w[:, feet_cfg.body_ids, 2]
    # Clearance above the current lower foot (matches profile extraction).
    support = torch.minimum(foot_z[:, 0], foot_z[:, 1])
    clear_l = foot_z[:, 0] - support
    clear_r = foot_z[:, 1] - support
    err = torch.square(clear_l - target_l) + torch.square(clear_r - target_r)
    reward = torch.exp(-err / (std * std))
    return reward * turning.float()


def undesired_contacts(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  return torch.sum((force > threshold).float(), dim=1)


def _foot_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  return torch.linalg.vector_norm(sensor.data.force_history, dim=-1).mean(dim=2)


def _stable_stand_weight(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  torso_cfg: SceneEntityCfg,
  command_name: str,
  command_threshold: float,
  planar_speed_scale: float,
  torso_ang_vel_scale: float,
  contact_force_threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene[torso_cfg.name]
  command = _command(env, command_name)
  command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
    command[:, 2]
  )
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
  motion_weight *= torch.exp(-torch.square(torso_ang_speed / torso_ang_vel_scale))
  return (
    (command_magnitude < command_threshold).float()
    * double_support.float()
    * motion_weight
  )


def stable_stand_hip_roll_norm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return torch.linalg.vector_norm(error, dim=1) * weight


def stable_stand_feet_fore_aft_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  position_deadband: float = 0.02,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  error = torch.sum(torch.relu(torch.abs(feet[:, :, 0]) - position_deadband), dim=1)
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return error * weight


def stable_stand_feet_lateral_center_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  center_deadband: float = 0.01,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  center = 0.5 * (feet[:, 0, 1] + feet[:, 1, 1])
  error = torch.relu(torch.abs(center) - center_deadband)
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return error * weight


def feet_support_roll_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  sensor_name: str,
  velocity_command_name: str = "twist",
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  quat = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
  gravity = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, len(feet_cfg.body_ids), -1)
  roll_error = torch.square(quat_apply_inverse(quat, gravity)[:, :, 1])
  support = (_foot_force(env, sensor_name) > contact_force_threshold).float()
  return (
    torch.sum(roll_error * support, dim=1)
    * moving_mask(env, velocity_command_name).float()
  )


def ankle_roll_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.linalg.vector_norm(error, dim=1)


def ankle_roll_action_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str,
  contact_force_threshold: float = 20.0,
  velocity_command_name: str = "twist",
) -> torch.Tensor:
  delta = (
    env.action_manager.action[:, asset_cfg.joint_ids]
    - env.action_manager.prev_action[:, asset_cfg.joint_ids]
  )
  loaded = (_foot_force(env, sensor_name) > contact_force_threshold).float()
  return (
    torch.sum(torch.square(delta) * loaded, dim=1)
    * moving_mask(env, velocity_command_name).float()
  )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions (same definition as tracking/kick)."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1).float()
