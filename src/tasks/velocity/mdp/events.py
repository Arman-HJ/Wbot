"""Velocity-task events (body-frame helpers on top of mjlab)."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_VEL_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


_STANDING_RECOVERY_PUSH_ATTR = "_standing_recovery_push_pending"
_STANDING_PUSH_RECOVERY_REWARD_NAME = "standing_push_recovery"


def _standing_push_recovery_reward_params(env: ManagerBasedRlEnv) -> dict:
  """Read push-gating params from the standing_push_recovery reward term."""
  return env.reward_manager.get_term_cfg(_STANDING_PUSH_RECOVERY_REWARD_NAME).params


def _standing_recovery_push_buf(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env flag set by body-frame pushes; consumed by standing recovery reward."""
  if not hasattr(env, _STANDING_RECOVERY_PUSH_ATTR):
    setattr(
      env,
      _STANDING_RECOVERY_PUSH_ATTR,
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
  return getattr(env, _STANDING_RECOVERY_PUSH_ATTR)


def mark_standing_recovery_push(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  push_delta_b: torch.Tensor,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  min_lin_impulse: float = 0.5,
  min_ang_impulse: float = 0.3,
) -> None:
  """Mark envs pushed while standing, only if the body-frame impulse is large enough.

  ``push_delta_b`` is the sampled root velocity delta in the body frame, shape
  ``[len(env_ids), 6]`` (lin xyz, ang roll/pitch/yaw). A push qualifies when
  ``||delta_lin|| >= min_lin_impulse`` **or** ``||delta_ang|| >= min_ang_impulse``.
  """
  pending = _standing_recovery_push_buf(env)
  lin_mag = torch.norm(push_delta_b[:, :3], dim=1)
  ang_mag = torch.norm(push_delta_b[:, 3:6], dim=1)
  large_push = (lin_mag >= min_lin_impulse) | (ang_mag >= min_ang_impulse)

  command = env.command_manager.get_command(command_name)
  if command is None:
    pending[env_ids] = large_push
    return
  cmd_speed = torch.norm(command[env_ids, :2], dim=1) + torch.abs(command[env_ids, 2])
  standing = cmd_speed <= command_threshold
  pending[env_ids] = standing & large_push



def push_by_setting_velocity_body_frame(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  flag_standing_recovery: bool = False,
) -> None:
  """Add root velocity impulse in the **body** frame (aligned with twist / IMU axes).

  ``velocity_range`` keys match ``push_by_setting_velocity``, but ``x/y/z`` and
  ``roll/pitch/yaw`` are interpreted in the root (pelvis) body frame before
  conversion to world frame for ``write_root_link_velocity_to_sim``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  quat_w = asset.data.root_link_quat_w[env_ids]

  range_list = [
    velocity_range.get(key, (0.0, 0.0)) for key in _VEL_KEYS
  ]
  ranges = torch.tensor(range_list, device=env.device)
  delta_b = sample_uniform(
    ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device
  )

  if flag_standing_recovery:
    mark_standing_recovery_push(env, env_ids, push_delta_b=delta_b)

  vel_w = vel_w.clone()
  vel_w[:, :3] += quat_apply(quat_w, delta_b[:, :3])
  vel_w[:, 3:6] += quat_apply(quat_w, delta_b[:, 3:6])
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_joints_by_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints by scaling default position/velocity (IsaacLab / engineai PM01)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  joint_ids = asset_cfg.joint_ids
  joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
  joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()
  joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_pos_limits = soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
