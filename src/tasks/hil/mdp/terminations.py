"""Mode-gated HIL terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.tasks.amp_v2.mdp.motion_loader import _AMP_MOTION_RESET_ATTR, AmpMotionResetLibrary
from src.tasks.hil.mdp.env_mode import hil_amp_mask, hil_track_mask
from mjlab.envs.mdp.terminations import bad_orientation, root_height_below_minimum

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def _motion_frame_indices(env: ManagerBasedRlEnv) -> torch.Tensor:
  twist = env.command_manager.get_term("twist")
  if hasattr(twist, "motion_frame_idx"):
    return twist.motion_frame_idx
  return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def hil_track_bad_joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  """Terminate track envs when any joint angle deviates too far from reference."""
  library: AmpMotionResetLibrary | None = getattr(env, _AMP_MOTION_RESET_ATTR, None)
  if library is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
  frame_idx = _motion_frame_indices(env)
  _, _, ref_joint_pos, _, _, _ = library.gather_reference_states(frame_idx)
  robot_joint_pos = asset.data.joint_pos[:, joint_ids]
  max_error = torch.max(torch.abs(robot_joint_pos - ref_joint_pos), dim=-1).values
  return (max_error > threshold) & hil_track_mask(env)


def hil_amp_fell_over(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  bad = bad_orientation(env, limit_angle, asset_cfg)
  return bad & hil_amp_mask(env)


def hil_amp_root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  bad = root_height_below_minimum(env, minimum_height, asset_cfg)
  return bad & hil_amp_mask(env)
