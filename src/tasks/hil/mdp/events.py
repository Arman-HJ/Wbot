"""HIL startup/reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from src.tasks.amp_v2.mdp.events import (
  init_amp_motion_reset_library,
)
from src.tasks.amp_v2.mdp.motion_loader import _AMP_MOTION_RESET_ATTR, AmpMotionResetLibrary
from src.tasks.hil.mdp.env_mode import MODE_AMP, MODE_TRACK, assign_hil_env_modes, get_hil_mode, init_hil_env_modes
from src.tasks.hil.mdp.twist_command import HilTwistCommand, sync_hil_twist_from_motion_frames

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def init_hil_training(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  track_fraction: float = 0.5,
  **init_motion_kwargs,
) -> None:
  """Startup: motion library + HIL mode assignment."""
  init_amp_motion_reset_library(env, env_ids, **init_motion_kwargs)
  init_hil_env_modes(env, env_ids=None, track_fraction=track_fraction)


def reset_hil_from_amp_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  joint_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  joint_position_range: tuple[float, float] = (0.0, 0.0),
  track_fraction: float = 0.5,
  psi_pose_range: dict[str, tuple[float, float]] | None = None,
  psi_velocity_range: dict[str, tuple[float, float]] | None = None,
  psi_joint_position_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
  """Reset robot from expert motion, assign HIL modes, PSI on track envs."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  assign_hil_env_modes(env, env_ids, track_fraction=track_fraction)

  library: AmpMotionResetLibrary | None = getattr(env, _AMP_MOTION_RESET_ATTR, None)
  if library is None:
    raise RuntimeError("AMP motion library not initialized.")

  asset: Entity = env.scene[asset_cfg.name]
  if asset.is_fixed_base:
    raise ValueError(
      f"reset_hil_from_amp_motion does not support fixed-base entity '{asset_cfg.name}'."
    )

  joint_ids, _ = asset.find_joints(
    joint_asset_cfg.joint_names, preserve_order=True
  )

  frame_indices, root_pos, root_quat, joint_pos, joint_vel, root_lin_vel, root_ang_vel = (
    library.sample(len(env_ids))
  )
  sync_hil_twist_from_motion_frames(env, env_ids, frame_indices)

  root_pos = root_pos + env.scene.env_origins[env_ids]

  if pose_range:
    range_list = [
      pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    root_pos = root_pos + pose_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_quat = quat_mul(orientations_delta, root_quat)

  if velocity_range:
    range_list = [
      velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    root_lin_vel = root_lin_vel + vel_samples[:, :3]
    root_ang_vel = root_ang_vel + vel_samples[:, 3:]

  if joint_position_range != (0.0, 0.0):
    joint_pos = joint_pos + sample_uniform(
      joint_position_range[0],
      joint_position_range[1],
      joint_pos.shape,
      device=env.device,
    )

  mode = get_hil_mode(env)
  track_local = (mode[env_ids] == MODE_TRACK).nonzero(as_tuple=False).flatten()
  if len(track_local) > 0 and (
    psi_pose_range or psi_velocity_range or psi_joint_position_range != (0.0, 0.0)
  ):
    _apply_psi(
      track_local,
      root_pos,
      root_quat,
      joint_pos,
      root_lin_vel,
      root_ang_vel,
      psi_pose_range,
      psi_velocity_range,
      psi_joint_position_range,
      env.device,
    )

  soft_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = torch.clip(joint_pos, soft_limits[..., 0], soft_limits[..., 1])

  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
  default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
  default_joint_pos[:, joint_ids] = joint_pos
  default_joint_vel[:, joint_ids] = joint_vel

  asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

  root_state = torch.cat(
    [root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1
  )
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
  asset.reset(env_ids=env_ids)

  amp_ids = env_ids[mode[env_ids] == MODE_AMP]
  twist = env.command_manager.get_term("twist")
  if isinstance(twist, HilTwistCommand) and len(amp_ids) > 0:
    twist.resample_amp_command(amp_ids)


def _apply_psi(
  track_local: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  psi_pose_range: dict[str, tuple[float, float]] | None,
  psi_velocity_range: dict[str, tuple[float, float]] | None,
  psi_joint_position_range: tuple[float, float],
  device: str,
) -> None:
  n = len(track_local)
  if psi_pose_range:
    range_list = [
      psi_pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=device)
    pose_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=device)
    root_pos[track_local] = root_pos[track_local] + pose_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_quat[track_local] = quat_mul(orientations_delta, root_quat[track_local])

  if psi_velocity_range:
    range_list = [
      psi_velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=device)
    vel_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=device)
    root_lin_vel[track_local] = root_lin_vel[track_local] + vel_samples[:, :3]
    root_ang_vel[track_local] = root_ang_vel[track_local] + vel_samples[:, 3:]

  if psi_joint_position_range != (0.0, 0.0):
    joint_pos[track_local] = joint_pos[track_local] + sample_uniform(
      psi_joint_position_range[0],
      psi_joint_position_range[1],
      joint_pos[track_local].shape,
      device=device,
    )
