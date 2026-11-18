"""Unitree-specific velocity rewards not in mjlab.tasks.velocity.mdp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.envs.mdp.rewards import action_acc_l2, action_rate_l2
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  wrap_to_pi,
  yaw_quat,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_yaw_frame_exp(
  env: ManagerBasedRlEnv,
  sigma: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  stand_threshold: float = 0.06,
) -> torch.Tensor:
  """Track linear velocity (xy) in yaw-aligned frame (engineai PM01).

  Standing commands use an L1 exponential kernel; walking uses L2.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stand_command = (torch.norm(command[:, :2], dim=1) < stand_threshold) & (
    torch.abs(command[:, 2]) < stand_threshold
  )
  asset: Entity = env.scene[asset_cfg.name]
  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
  )
  lin_vel_error_square = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
  lin_vel_error_abs = torch.sum(torch.abs(command[:, :2] - vel_yaw[:, :2]), dim=1)
  rew_square = torch.exp(-lin_vel_error_square * sigma)
  rew_abs = torch.exp(-lin_vel_error_abs * sigma)
  return torch.where(stand_command, rew_abs, rew_square)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  stand_threshold: float = 0.06,
) -> torch.Tensor:
  """Track yaw angular velocity in world frame (engineai PM01).

  Standing commands use an L1 exponential kernel; walking uses L2.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stand_command = (torch.norm(command[:, :2], dim=1) < stand_threshold) & (
    torch.abs(command[:, 2]) < stand_threshold
  )
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_error_square = torch.square(
    command[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
  )
  ang_vel_error_abs = torch.abs(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  rew_square = torch.exp(-ang_vel_error_square * sigma)
  rew_abs = torch.exp(-ang_vel_error_abs * sigma)
  return torch.where(stand_command, rew_abs, rew_square)


def base_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping base roll/pitch near zero (engineai PM01)."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return torch.exp(-torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1) * 10.0)


def track_base_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "base_height",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_sensor_name: str | None = "terrain_scan",
) -> torch.Tensor:
  """Reward tracking the commanded base height above local terrain."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    body_z = asset.data.root_link_pos_w[:, 2]
  ground_z = torch.zeros_like(body_z)
  if terrain_sensor_name is not None:
    try:
      ground_z = _ground_height_from_terrain_scan(env, terrain_sensor_name)
    except KeyError:
      pass
  error = torch.square(command.squeeze(-1) - (body_z - ground_z))
  return torch.exp(-error / std**2)


def base_height_tracking(
  env: ManagerBasedRlEnv,
  target_height: float = 0.82,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping root height near a target (engineai PM01)."""
  asset: Entity = env.scene[asset_cfg.name]
  height_error = torch.abs(asset.data.root_link_pos_w[:, 2] - target_height)
  return torch.exp(-height_error * 30.0)


def feet_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  stand_threshold: float = 0.06,
) -> torch.Tensor:
  """Reward aligning feet orientation; ignore yaw error while turning (engineai PM01)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  yaw_command = torch.abs(command[:, 2]) > stand_threshold

  asset: Entity = env.scene[asset_cfg.name]
  feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  base_quat = asset.data.root_link_quat_w

  num_envs, num_feet, _ = feet_quat.shape
  feet_flat = feet_quat.reshape(-1, 4)
  roll, pitch, yaw = euler_xyz_from_quat(feet_flat)
  roll = roll.reshape(num_envs, num_feet)
  pitch = pitch.reshape(num_envs, num_feet)
  yaw = yaw.reshape(num_envs, num_feet)

  _, _, base_yaw = euler_xyz_from_quat(base_quat)

  feet_roll_pitch_error = torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1)
  feet_yaw_error = torch.abs(wrap_to_pi(yaw - base_yaw.unsqueeze(1)))

  rew = torch.sum(feet_roll_pitch_error + feet_yaw_error, dim=1)
  rew[yaw_command] = torch.sum(feet_roll_pitch_error[yaw_command], dim=1)
  return torch.exp(-rew * 2.0)


def joint_deviation_exp(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  tolerance: float = 0.0,
  scale: float = 3.0,
  max_err: float = 50.0,
) -> torch.Tensor:
  """Reward keeping selected joints near their default pose (engineai PM01)."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  joint_error = torch.norm(joint_pos - default_pos, dim=1)
  joint_error = torch.clamp(joint_error - tolerance, min=0.0, max=max_err)
  return torch.exp(-joint_error * scale)


def energy_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize approximate energy consumption (|torque * velocity| sum)."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_torques = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_torques * joint_vel), dim=1)


def _epoch_curriculum_scale(
  env: ManagerBasedRlEnv,
  start_scale: float,
  power: float,
  interval_epochs: int,
) -> float:
  """Scale penalty by ``start_scale ** (power ** updates)`` every interval_epochs."""
  interval = max(int(interval_epochs), 1)
  updates = env.common_step_counter // interval
  return float(start_scale) ** (float(power) ** updates)


def energy_cost_with_curriculum(
  env: ManagerBasedRlEnv,
  start_scale: float,
  power: float,
  interval_epochs: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Energy cost with epoch-based curriculum scaling (engineai PM01)."""
  reward = energy_cost(env, asset_cfg=asset_cfg)
  return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def action_rate_with_curriculum(
  env: ManagerBasedRlEnv,
  start_scale: float,
  power: float,
  interval_epochs: int,
) -> torch.Tensor:
  """Action rate penalty with epoch-based curriculum scaling (engineai PM01)."""
  reward = action_rate_l2(env)
  return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def action_smoothness_with_curriculum(
  env: ManagerBasedRlEnv,
  start_scale: float,
  power: float,
  interval_epochs: int,
) -> torch.Tensor:
  """Action smoothness penalty with epoch-based curriculum scaling (engineai PM01)."""
  reward = action_acc_l2(env)
  return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def variable_posture(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  std_standing: dict[str, float],
  std_walking: dict[str, float],
  std_running: dict[str, float],
  walking_threshold: float = 0.1,
  running_threshold: float = 2.0,
) -> torch.Tensor:
  """Speed-dependent posture reward with pose-error logging metrics."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  _, joint_names = asset.find_joints(asset_cfg.joint_names)
  _, _, std_standing_vals = resolve_matching_names_values(
    data=std_standing,
    list_of_strings=joint_names,
  )
  _, _, std_walking_vals = resolve_matching_names_values(
    data=std_walking,
    list_of_strings=joint_names,
  )
  _, _, std_running_vals = resolve_matching_names_values(
    data=std_running,
    list_of_strings=joint_names,
  )
  std_standing_t = torch.tensor(std_standing_vals, device=env.device, dtype=torch.float32)
  std_walking_t = torch.tensor(std_walking_vals, device=env.device, dtype=torch.float32)
  std_running_t = torch.tensor(std_running_vals, device=env.device, dtype=torch.float32)

  speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = speed < walking_threshold
  running = speed >= running_threshold
  walking = (~standing) & (~running)

  std = (
    std_standing_t * standing.float().unsqueeze(1)
    + std_walking_t * walking.float().unsqueeze(1)
    + std_running_t * running.float().unsqueeze(1)
  )
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error_sq = torch.square(current - default)
  pose_error = torch.mean(error_sq, dim=1)
  reward = torch.exp(-torch.mean(error_sq / (std**2), dim=1))

  log = env.extras.setdefault("log", {})
  if torch.any(standing):
    log["Metrics/pose_error_standing_mean"] = torch.mean(pose_error[standing])
  if torch.any(walking):
    log["Metrics/pose_error_walking_mean"] = torch.mean(pose_error[walking])
  if torch.any(running):
    log["Metrics/pose_error_running_mean"] = torch.mean(pose_error[running])
  return reward


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  leg_phase = (global_phase + offsets) % 1.0
  is_stance = leg_phase < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      reward *= active
  return reward


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command <= command_threshold).float()
      reward *= scale
  return reward


def both_feet_air(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize simultaneous flight of all tracked feet (double-foot jump).

  Returns 1 when every foot in ``sensor_name`` is off the ground, else 0.
  Use with a negative reward weight.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  in_contact = contact_sensor.data.current_contact_time > 0.0
  return torch.all(~in_contact, dim=1).float()


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize L1 deviation of selected joints from the default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(angle), dim=1)


def joint_pos_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize squared deviation of selected joints from default (initial) pose."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  cost = torch.sum(torch.square(diff), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost *= (total_command <= command_threshold).float()
  return cost


class standing_push_two_step_recovery:
  """Reward minimal stepping to recover balance after a standing body-frame push."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.command_name = cfg.params["command_name"]
    self.command_threshold = cfg.params.get("command_threshold", 0.1)
    self.min_lin_impulse = float(cfg.params.get("min_lin_impulse", 0.5))
    self.min_ang_impulse = float(cfg.params.get("min_ang_impulse", 0.3))
    self.target_steps = int(cfg.params.get("target_steps", 2))
    self.recovery_time_s = float(cfg.params.get("recovery_time_s", 2.5))
    self.lin_vel_threshold = float(cfg.params.get("lin_vel_threshold", 0.12))
    self.ang_vel_threshold = float(cfg.params.get("ang_vel_threshold", 0.45))
    self.upright_threshold = float(cfg.params.get("upright_threshold", 0.12))
    self.step_dt = env.step_dt
    self.max_recovery_steps = max(1, int(self.recovery_time_s / self.step_dt))

    n = env.num_envs
    dev = env.device
    self.in_recovery = torch.zeros(n, dtype=torch.bool, device=dev)
    self.landing_count = torch.zeros(n, dtype=torch.int32, device=dev)
    self.recovery_age = torch.zeros(n, dtype=torch.int32, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.in_recovery[env_ids] = False
    self.landing_count[env_ids] = 0
    self.recovery_age[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    min_lin_impulse: float = 0.5,
    min_ang_impulse: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_steps: int = 2,
    recovery_time_s: float = 2.5,
    lin_vel_threshold: float = 0.12,
    ang_vel_threshold: float = 0.45,
    upright_threshold: float = 0.12,
  ) -> torch.Tensor:
    del (
      sensor_name,
      command_threshold,
      min_lin_impulse,
      min_ang_impulse,
      target_steps,
      recovery_time_s,
      lin_vel_threshold,
      ang_vel_threshold,
      upright_threshold,
    )

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    command = env.command_manager.get_command(command_name)
    if command is None:
      return reward

    cmd_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = cmd_speed <= self.command_threshold

    pending_attr = "_standing_recovery_push_pending"
    if hasattr(env, pending_attr):
      pushed = getattr(env, pending_attr) & standing
      getattr(env, pending_attr)[:] = False
      start = pushed
      self.in_recovery |= start
      self.landing_count[start] = 0
      self.recovery_age[start] = 0

    active = self.in_recovery & standing
    self.in_recovery &= standing

    if not torch.any(active):
      return reward

    contact_sensor: ContactSensor = env.scene[self.sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    landed = first_contact.any(dim=1)
    self.landing_count[active] += landed[active].to(torch.int32)
    self.recovery_age[active] += 1

    asset: Entity = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_b
    ang_vel = asset.data.root_link_ang_vel_b
    grav_xy = asset.data.projected_gravity_b[:, :2]
    in_contact = contact_sensor.data.current_contact_time > 0.0
    both_feet = torch.all(in_contact, dim=1)

    stable = (
      (torch.norm(lin_vel[:, :2], dim=1) < self.lin_vel_threshold)
      & (torch.norm(ang_vel, dim=1) < self.ang_vel_threshold)
      & (torch.norm(grav_xy, dim=1) < self.upright_threshold)
      & both_feet
    )
    stable_active = stable & active

    shaping = torch.exp(-torch.sum(torch.square(grav_xy), dim=1) / self.upright_threshold**2)
    reward[active] = 0.05 * shaping[active]

    extra = active & (self.landing_count > self.target_steps)
    reward[extra] -= 0.15 * (
      self.landing_count[extra].to(torch.float32) - float(self.target_steps)
    )

    if torch.any(stable_active):
      err = self.landing_count[stable_active].to(torch.float32) - float(self.target_steps)
      reward[stable_active] += torch.exp(-torch.square(err) / 0.5)
      self.in_recovery[stable_active] = False

    timed_out = active & (self.recovery_age >= self.max_recovery_steps)
    reward[timed_out] -= 0.25
    self.in_recovery[timed_out] = False

    n_rec = torch.sum(active).float()
    env.extras["log"]["Metrics/standing_recovery_active_frac"] = n_rec / env.num_envs
    if torch.any(stable_active):
      env.extras["log"]["Metrics/standing_recovery_steps_at_stable"] = torch.mean(
        self.landing_count[stable_active].float()
      )
    return reward


class upright:
  """Upright reward with optional ``gravity_sensor_name`` (overrides stale mjlab installs).

  Matches :class:`mjlab.tasks.velocity.mdp.rewards.upright` so PND can use
  ``robot/torso_upvector`` aligned with ``projected_gravity_from_sensor``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._gravity_sensor_name: str | None = cfg.params.get("gravity_sensor_name")
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
    gravity_sensor_name: str | None = None,
  ) -> torch.Tensor:
    del gravity_sensor_name
    asset: Entity = env.scene[asset_cfg.name]

    if terrain_sensor_names is not None:
      if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        body_quat_w = body_quat_w.squeeze(1)
      else:
        body_quat_w = asset.data.root_link_quat_w
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    elif self._gravity_sensor_name is not None:
      upvector_sensor: BuiltinSensor = env.scene[self._gravity_sensor_name]
      projected_gravity_b = -upvector_sensor.data
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
      if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        body_quat_w = body_quat_w.squeeze(1)
      else:
        body_quat_w = asset.data.root_link_quat_w
      gravity_w = asset.data.gravity_vec_w
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]
    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    if self._gravity_sensor_name is not None:
      upvector_sensor: BuiltinSensor = env.scene[self._gravity_sensor_name]
      body_up_w = quat_apply(body_quat_w, -upvector_sensor.data)
    else:
      up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
        body_quat_w[:, :3]
      )
      body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


upright_from_gravity_sensor = upright



class standing_waist_settle_exp:
  """Standing-only torso settling reward using only built-in sensors."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.command_name = cfg.params["command_name"]
    self.command_threshold = float(cfg.params.get("command_threshold", 0.1))
    self.angvel_sensor_name = cfg.params.get("angvel_sensor_name", "robot/imu_torso_ang_vel")
    self.gravity_sensor_name = cfg.params.get("gravity_sensor_name", "robot/imu_upvector")
    self.w_omega = float(cfg.params.get("w_omega", 1.0))
    self.w_tilt = float(cfg.params.get("w_tilt", 1.5))
    self.std_delta = float(cfg.params.get("std_delta", 0.08))
    self.k_fast = float(cfg.params.get("k_fast", 0.05))
    self.std_fast = float(cfg.params.get("std_fast", 1.0))
    self.step_dt = float(env.step_dt)

    n = env.num_envs
    dev = env.device
    self.prev_err = torch.zeros(n, device=dev, dtype=torch.float32)
    self.initialized = torch.zeros(n, device=dev, dtype=torch.bool)
    self._missing_sensor_warned = False

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.prev_err[env_ids] = 0.0
    self.initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    angvel_sensor_name: str = "robot/imu_torso_ang_vel",
    gravity_sensor_name: str = "robot/imu_upvector",
    w_omega: float = 1.0,
    w_tilt: float = 1.5,
    std_delta: float = 0.08,
    k_fast: float = 0.05,
    std_fast: float = 1.0,
  ) -> torch.Tensor:
    del (
      command_name,
      command_threshold,
      angvel_sensor_name,
      gravity_sensor_name,
      w_omega,
      w_tilt,
      std_delta,
      k_fast,
      std_fast,
    )
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    command = env.command_manager.get_command(self.command_name)
    if command is None:
      return reward

    try:
      angvel_sensor: BuiltinSensor = env.scene[self.angvel_sensor_name]
      upvector_sensor: BuiltinSensor = env.scene[self.gravity_sensor_name]
    except KeyError:
      if not self._missing_sensor_warned:
        env.extras["log"]["Metrics/standing_waist_settle_missing_sensor"] = 1.0
        self._missing_sensor_warned = True
      return reward

    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = total_command <= self.command_threshold

    omega = angvel_sensor.data
    projected_gravity = -upvector_sensor.data
    err = (
      self.w_omega * torch.sum(torch.square(omega), dim=1)
      + self.w_tilt * torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    )
    improve = torch.relu(self.prev_err - err)
    improve = torch.where(self.initialized, improve, torch.zeros_like(improve))
    delta = torch.relu(err - self.prev_err)
    delta = torch.where(self.initialized, delta, torch.zeros_like(delta))

    non_worsening = torch.exp(-delta / (self.std_delta**2))
    improve_rate = improve / max(self.step_dt, 1e-6)
    fast_settle = self.k_fast * (1.0 - torch.exp(-improve_rate / max(self.std_fast, 1e-6)))
    reward = standing.float() * (non_worsening + fast_settle)

    if torch.any(standing):
      env.extras["log"]["Metrics/standing_waist_err_mean"] = torch.mean(err[standing])
      env.extras["log"]["Metrics/standing_waist_delta_mean"] = torch.mean(delta[standing])
      env.extras["log"]["Metrics/standing_waist_improve_mean"] = torch.mean(improve[standing])

    self.prev_err = err.detach()
    self.initialized[:] = True
    return reward


def _standing_command_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Mask active when the velocity command is near zero (standing)."""
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  return (total_command <= command_threshold).float()


def stand_joint_pos_l1(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize L1 joint position deviation from default when command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )
  return error * _standing_command_mask(env, command_name, command_threshold)


def stand_joint_vel_l2(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize L2 joint velocity when command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  return error * _standing_command_mask(env, command_name, command_threshold)


def stand_action_rate_l2(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize action rate only while standing (anti-shake / chatter)."""
  return action_rate_l2(env) * _standing_command_mask(
    env, command_name, command_threshold
  )


def _walking_command_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Mask active when the velocity command is above threshold (walking/running)."""
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (total_command > command_threshold).float()


def arm_swing_coordination(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  left_hip_pitch_idx: int = 0,
  right_hip_pitch_idx: int = 6,
  left_shoulder_pitch_idx: int = 15,
  right_shoulder_pitch_idx: int = 22,
  contralateral_weight: float = 1.0,
  ipsilateral_weight: float = 1.0,
) -> torch.Tensor:
  """Reward contralateral arm–leg swing using signed hip/shoulder pitch velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel

  v_l_hip = joint_vel[:, left_hip_pitch_idx]
  v_r_hip = joint_vel[:, right_hip_pitch_idx]
  v_l_arm = joint_vel[:, left_shoulder_pitch_idx]
  v_r_arm = joint_vel[:, right_shoulder_pitch_idx]

  contralateral_coord = torch.exp(
    -torch.square(v_r_hip - v_l_arm) - torch.square(v_l_hip - v_r_arm)
  )
  ipsilateral_penalty = torch.exp(
    -(torch.square(v_r_hip - v_r_arm) + torch.square(v_l_hip - v_l_arm))
  )
  coordination_reward = (
    contralateral_weight * contralateral_coord
    - ipsilateral_weight * ipsilateral_penalty
  )
  return coordination_reward * _walking_command_mask(
    env, command_name, command_threshold
  )


def _ground_height_from_terrain_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Median terrain hit height under the raycast sensor frame."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, RayCastSensor):
    raise TypeError(
      f"base_height_exp terrain sensor must be RayCastSensor, got {type(sensor).__name__}"
    )
  hit_z = sensor.data.hit_pos_w[..., 2]
  valid = sensor.data.distances >= 0
  masked = torch.where(valid, hit_z, torch.full_like(hit_z, float("nan")))
  ground_z = torch.nanmedian(masked, dim=1).values
  return torch.nan_to_num(ground_z, nan=0.0)


def base_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_sensor_name: str | None = "terrain_scan",
) -> torch.Tensor:
  """Reward maintaining target clearance height above local terrain."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    body_z = asset.data.root_link_pos_w[:, 2]

  if terrain_sensor_name is not None:
    try:
      ground_z = _ground_height_from_terrain_scan(env, terrain_sensor_name)
    except KeyError:
      ground_z = torch.zeros_like(body_z)
  else:
    ground_z = torch.zeros_like(body_z)

  height_err_sq = torch.square(body_z - ground_z - target_height)
  return torch.exp(-height_err_sq / std**2)


def feet_distance(
  env: ManagerBasedRlEnv,
  target_dist: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  std: float = 0.04,
) -> torch.Tensor:
  """Reward heading-frame lateral foot spacing toward a fixed ``target_dist``.

  Uses ``y_left - y_right`` in the yaw-aligned base frame (not xy norm), so a
  long stride cannot mask a narrow or cross-legged stance.
  ``asset_cfg.body_names`` must be ordered ``(left, right)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :3]
  assert feet_pos_w.shape[1] == 2, "feet_distance expects exactly two feet (left, right)"

  base_pos_w = asset.data.root_link_pos_w
  yaw_q = yaw_quat(asset.data.root_link_quat_w)
  feet_rel = feet_pos_w - base_pos_w.unsqueeze(1)
  num_envs = feet_rel.shape[0]
  yaw_q_flat = yaw_q.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
  feet_h = quat_apply_inverse(yaw_q_flat, feet_rel.reshape(-1, 3)).reshape(
    num_envs, 2, 3
  )

  # Positive when left is on +Y and right on -Y (not crossed).
  lateral = feet_h[:, 0, 1] - feet_h[:, 1, 1]
  error = lateral - target_dist
  reward = torch.exp(-torch.square(error) / (std**2))

  log = env.extras.setdefault("log", {})
  log["Metrics/feet_distance_mean"] = torch.mean(lateral)
  log["Metrics/feet_distance_err_mean"] = torch.mean(torch.abs(error))
  log["Metrics/feet_cross_frac"] = torch.mean((lateral <= 0.0).float())

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward = reward * (total_command > command_threshold).float()

  return reward


def feet_position(
  env: ManagerBasedRlEnv,
  command_name: str | None = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  stand_threshold: float | None = 0.1,
  ankle_distance: float = 0.22,
  base_height_target: float = 0.82,
) -> torch.Tensor:
  """Reward keeping feet near a desired heading-frame stance.

  If ``stand_threshold`` is set, only apply when the velocity command is below
  that threshold (else return 1). Pass ``stand_threshold=None`` to always apply.
  ``asset_cfg.body_names`` must be ordered (left, right).
  """
  asset: Entity = env.scene[asset_cfg.name]

  feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  base_pos_w = asset.data.root_link_pos_w
  base_quat_w = asset.data.root_link_quat_w

  r, p, y = euler_xyz_from_quat(base_quat_w)
  heading_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
  feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
  num_envs, num_feet, _ = feet_pos_rel.shape
  heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
  feet_pos_heading = quat_apply_inverse(
    heading_quat_per_foot, feet_pos_rel.reshape(-1, 3)
  ).reshape(num_envs, num_feet, 3)

  desired_x = torch.zeros((num_envs, num_feet), device=feet_pos_heading.device)
  half = num_feet // 2
  desired_y = torch.cat(
    (
      (ankle_distance * 0.5)
      * torch.ones((num_envs, half), device=feet_pos_heading.device),
      (-ankle_distance * 0.5)
      * torch.ones((num_envs, num_feet - half), device=feet_pos_heading.device),
    ),
    dim=1,
  )
  desired_z = -(base_height_target - 0.045) * torch.ones(
    (num_envs, num_feet), device=feet_pos_heading.device
  )
  desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)

  position_error = torch.sum(torch.abs(feet_pos_heading - desired), dim=(1, 2))
  reward_stand = torch.exp(-position_error * 3.0)

  if stand_threshold is None:
    return reward_stand

  assert command_name is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stand_command = (torch.norm(command[:, :2], dim=1) < stand_threshold) & (
    torch.abs(command[:, 2]) < stand_threshold
  )
  return torch.where(stand_command, reward_stand, torch.ones_like(reward_stand))


def _foot_contact_flags(
  contact_sensor: ContactSensor,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Return per-foot contact flags [B, N].

  MuJoCo ``netforce`` reports world-frame wrenches whose z sign may be negative
  on foot–terrain contacts, so do not use ``force[..., 2] > threshold`` alone.
  Prefer the sensor ``found`` field; fall back to force magnitude.
  """
  data = contact_sensor.data
  if data.found is not None:
    return data.found > 0
  forces = data.force
  assert forces is not None
  return torch.norm(forces, dim=-1) > force_threshold


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tangential_ratio: float = 5.0,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize feet hitting vertical surfaces (IsaacLab / TienKung style).

  Triggers when ``||F_xy|| > tangential_ratio * |F_z|`` on a contacting foot.
  This ratio form (not absolute thresholds) is what actually fires during
  locomotion; the PM01 absolute form ``F_xy>2 & |F_z|<1`` is near-impossible
  on flat ground where stance |F_z| is hundreds of Newtons.

  Uses force history when available so brief impacts are not missed.
  Returns the count of stumbling feet per environment.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force_history = contact_sensor.data.force_history
  if force_history is not None:
    # [B, N, H, 3]
    fxy = torch.norm(force_history[..., :2], dim=-1)
    fz = torch.abs(force_history[..., 2])
    in_contact = torch.norm(force_history, dim=-1) > contact_force_threshold
    stumble = ((fxy > tangential_ratio * fz) & in_contact).any(dim=-1)
  else:
    forces = contact_sensor.data.force
    assert forces is not None
    fxy = torch.norm(forces[..., :2], dim=-1)
    fz = torch.abs(forces[..., 2])
    in_contact = torch.norm(forces, dim=-1) > contact_force_threshold
    stumble = (fxy > tangential_ratio * fz) & in_contact
  return stumble.sum(dim=1).float()


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize foot xy velocity while in contact (engineai / IsaacLab feet_slide)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  forces = contact_sensor.data.force
  assert forces is not None
  if contact_sensor.data.force_history is not None:
    contacts = (
      torch.norm(contact_sensor.data.force_history, dim=-1).max(dim=2)[0]
      > contact_force_threshold
    )
  else:
    contacts = torch.norm(forces, dim=-1) > contact_force_threshold

  asset: Entity = env.scene[asset_cfg.name]
  body_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  return torch.sum(body_vel.norm(dim=-1) * contacts.float(), dim=1)


def feet_contact_fixed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  stand_threshold: float = 0.1,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Reward valid foot contacts during walking and standing (engineai PM01).

  - Stand command: reward 1 only if both feet are in contact.
  - Walk command: reward 1 if any recent timestep had exactly one foot in contact.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stand_command = (torch.norm(command[:, :2], dim=1) < stand_threshold) & (
    torch.abs(command[:, 2]) < stand_threshold
  )

  contact_sensor: ContactSensor = env.scene[sensor_name]
  force_history = contact_sensor.data.force_history
  if force_history is not None:
    # [B, N, H, 3]; index 0 = most recent substep.
    contacts = torch.norm(force_history, dim=-1) > force_threshold
    contact_num_buf = torch.sum(contacts, dim=1)  # [B, H]
    stand_contact = contact_num_buf[:, 0] == 2
    single_contact = torch.any(contact_num_buf == 1, dim=1)
  else:
    contacts = _foot_contact_flags(contact_sensor, force_threshold)  # [B, N]
    contact_num = torch.sum(contacts, dim=1)
    stand_contact = contact_num == 2
    single_contact = contact_num == 1

  reward = (stand_command & stand_contact).float()
  reward = torch.where((~stand_command) & single_contact, torch.ones_like(reward), reward)
  return reward


def feet_air_time_on_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.5,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward long steps on first contact (IsaacLab / engineai ``feet_air_time``).

  Named separately from mjlab ``feet_air_time`` (current-air-time band reward).
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(env.step_dt)
  last_air_time = contact_sensor.data.last_air_time
  assert last_air_time is not None
  reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward *= (torch.norm(command[:, :2], dim=1) > command_threshold).float()
  return reward


def feet_air_time_similarity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  scale: float = 4.0,
  min_air_time: float = 0.0,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward similar completed air times between the two feet (anti-limp).

  Fires when either foot makes a new contact and both last air times are valid.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(env.step_dt)
  last_air_time = contact_sensor.data.last_air_time
  assert last_air_time is not None
  if last_air_time.shape[-1] < 2:
    raise ValueError("feet_air_time_similarity expects at least two feet in the contact sensor.")

  recent_contact = torch.any(first_contact[:, :2] > 0.0, dim=1)
  valid = torch.all(last_air_time[:, :2] > min_air_time, dim=1)
  diff = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
  reward = torch.exp(-diff * scale)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
  return reward * (recent_contact & valid).float() * active


def feet_air_time_positive_biped(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.5,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward single-stance mode time up to ``threshold`` (engineai dense air-time)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  air_time = contact_sensor.data.current_air_time
  contact_time = contact_sensor.data.current_contact_time
  assert air_time is not None and contact_time is not None
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)),
    dim=1,
  )[0]
  reward = torch.clamp(reward, max=threshold)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward *= (torch.norm(command[:, :2], dim=1) > command_threshold).float()
  return reward


def feet_lateral_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  local_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize foot yaw misalignment vs pelvis (prevent toe-out / 外八字)."""
  asset: Entity = env.scene[asset_cfg.name]
  feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  num_feet = feet_quat.shape[1]

  forward = torch.tensor(
    local_forward_axis, device=env.device, dtype=torch.float32
  )
  body_forward_w = quat_apply(
    asset.data.root_link_quat_w,
    forward.unsqueeze(0).expand(env.num_envs, -1),
  ).unsqueeze(1).expand(-1, num_feet, -1)

  feet_quat_flat = feet_quat.reshape(-1, 4)
  foot_forward_w = quat_apply(
    feet_quat_flat,
    forward.unsqueeze(0).expand(feet_quat_flat.shape[0], -1),
  ).reshape(env.num_envs, num_feet, 3)

  body_yaw = torch.atan2(body_forward_w[..., 1], body_forward_w[..., 0])
  foot_yaw = torch.atan2(foot_forward_w[..., 1], foot_forward_w[..., 0])
  yaw_diff = torch.atan2(
    torch.sin(foot_yaw - body_yaw), torch.cos(foot_yaw - body_yaw)
  )

  penalty = torch.sum(torch.square(yaw_diff), dim=1)

  log = env.extras.setdefault("log", {})
  log["Metrics/feet_yaw_deviation_mean"] = torch.mean(torch.abs(yaw_diff))

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      penalty = penalty * (total_command > command_threshold).float()

  return penalty


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gravity_sensor_name: str | None = None,
) -> torch.Tensor:
  """Penalize non-flat base orientation (L2 on projected gravity xy).

  If ``gravity_sensor_name`` is set, uses that upvector sensor (e.g. torso IMU)
  instead of FK from ``asset_cfg`` body/root orientation.
  """
  if gravity_sensor_name is not None:
    upvector_sensor: BuiltinSensor = env.scene[gravity_sensor_name]
    projected_gravity_b = -upvector_sensor.data
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared
