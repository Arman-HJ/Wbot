"""Mode-gated HIL rewards (tracking + velocity task)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from src.tasks.amp_v2.mdp.commands import AmpExpertMotionCommand
from src.tasks.amp_v2.mdp.motion_loader import _AMP_MOTION_RESET_ATTR, AmpMotionResetLibrary
from src.tasks.hil.mdp.env_mode import hil_amp_mask, hil_track_mask
from mjlab.tasks.velocity.mdp.rewards import (
  track_angular_velocity,
  track_linear_velocity,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _reference_root_state(
  env: ManagerBasedRlEnv, frame_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  library: AmpMotionResetLibrary | None = getattr(env, _AMP_MOTION_RESET_ATTR, None)
  if library is None:
    z = torch.zeros(env.num_envs, 3, device=env.device)
    q = torch.zeros(env.num_envs, 4, device=env.device)
    q[:, 0] = 1.0
    return z, q, z, z

  ref = library.loader.all_reference_states[frame_indices]
  joint_dim = (ref.shape[1] - 13) // 2
  off = 7 + 2 * joint_dim
  root_pos = ref[:, :3] + env.scene.env_origins
  root_quat = ref[:, 3:7]
  root_lin_vel = ref[:, off : off + 3]
  root_ang_vel = ref[:, off + 3 : off + 6]
  return root_pos, root_quat, root_lin_vel, root_ang_vel


def _motion_frame_indices(env: ManagerBasedRlEnv) -> torch.Tensor:
  twist = env.command_manager.get_term("twist")
  if hasattr(twist, "motion_frame_idx"):
    return twist.motion_frame_idx
  return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def hil_track_root_pos_reward(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  frame_idx = _motion_frame_indices(env)
  ref_pos, _, _, _ = _reference_root_state(env, frame_idx)
  error = torch.sum(torch.square(asset.data.root_link_pos_w - ref_pos), dim=-1)
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_track_root_rot_reward(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  frame_idx = _motion_frame_indices(env)
  _, ref_quat, _, _ = _reference_root_state(env, frame_idx)
  error = quat_error_magnitude(asset.data.root_link_quat_w, ref_quat) ** 2
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_track_root_lin_vel_reward(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  frame_idx = _motion_frame_indices(env)
  _, _, ref_lin_vel, _ = _reference_root_state(env, frame_idx)
  error = torch.sum(torch.square(asset.data.root_link_lin_vel_w - ref_lin_vel), dim=-1)
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_track_root_ang_vel_reward(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  frame_idx = _motion_frame_indices(env)
  _, _, _, ref_ang_vel = _reference_root_state(env, frame_idx)
  error = torch.sum(torch.square(asset.data.root_link_ang_vel_w - ref_ang_vel), dim=-1)
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_track_anchor_pos_reward(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "motion",
) -> torch.Tensor:
  command = cast(AmpExpertMotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_track_anchor_rot_reward(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "motion",
) -> torch.Tensor:
  command = cast(AmpExpertMotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  reward = torch.exp(-error / std**2)
  return reward * hil_track_mask(env).to(dtype=reward.dtype)


def hil_amp_track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  reward = track_linear_velocity(env, std, command_name, asset_cfg)
  return reward * hil_amp_mask(env).to(dtype=reward.dtype)


def hil_amp_track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  reward = track_angular_velocity(env, std, command_name, asset_cfg)
  return reward * hil_amp_mask(env).to(dtype=reward.dtype)
