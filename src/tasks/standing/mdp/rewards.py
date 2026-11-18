"""Standing task reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from src.tasks.standing.mdp.leg_pitch_ik import LegPitchIkParams, leg_pitch_ik_from_base_height
from src.tasks.standing.mdp.upper_body_motion import UpperBodyMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def tracking_upper_body_dofs(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Strict upper-body joint tracking (FALCON ``tracking_upper_body_dofs``)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(UpperBodyMotionCommand, env.command_manager.get_term(command_name))
  if command.fix_upper_body.all():
    return torch.ones(env.num_envs, device=env.device)

  joint_ids = asset_cfg.joint_ids
  if joint_ids is None:
    joint_ids, _ = asset.find_joints(
      list(command.cfg.joint_names), preserve_order=True
    )
  upper_pos = asset.data.joint_pos[:, joint_ids]
  error = torch.sum(torch.square(upper_pos - command.ref_joint_pos), dim=1)
  reward = torch.exp(-error / std**2)
  reward[command.fix_upper_body] = 1.0
  return reward


def _resolve_leg_pitch_ik_params(
  thigh_length: float | None,
  shank_length: float | None,
  torso_hip_offset: float | None,
) -> LegPitchIkParams:
  defaults = LegPitchIkParams()
  return LegPitchIkParams(
    thigh_length=float(thigh_length)
    if thigh_length is not None
    else defaults.thigh_length,
    shank_length=float(shank_length)
    if shank_length is not None
    else defaults.shank_length,
    torso_hip_offset=float(torso_hip_offset)
    if torso_hip_offset is not None
    else defaults.torso_hip_offset,
  )


def _apply_height_ik_to_reference(
  env: ManagerBasedRlEnv,
  asset: Entity,
  asset_cfg: SceneEntityCfg,
  height_command_name: str,
  thigh_length: float | None,
  shank_length: float | None,
  torso_hip_offset: float | None,
) -> torch.Tensor:
  """Build reference joints for ``asset_cfg`` with pitch IK where applicable."""
  assert asset_cfg.joint_ids is not None
  _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
  ref = asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone()

  height_cmd = env.command_manager.get_command(height_command_name)
  if height_cmd is None:
    return ref

  params = _resolve_leg_pitch_ik_params(
    thigh_length,
    shank_length,
    torso_hip_offset,
  )
  hip_ref, knee_ref, ankle_ref = leg_pitch_ik_from_base_height(
    height_cmd.squeeze(-1), params
  )

  pitch_patterns = (
    (r".*_hip_pitch_joint", hip_ref),
    (r".*_knee_joint", knee_ref),
    (r".*_ankle_pitch_joint", ankle_ref),
  )
  for pattern, values in pitch_patterns:
    ids, names = asset.find_joints(pattern, preserve_order=True)
    name_to_value = {name: values for name in names}
    for local_idx, name in enumerate(joint_names):
      if name in name_to_value:
        ref[:, local_idx] = name_to_value[name]
  return ref


def variable_posture_height_aware(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  height_command_name: str = "base_height",
  std_standing: dict[str, float] = {},
  std_walking: dict[str, float] = {},
  std_running: dict[str, float] = {},
  walking_threshold: float = 0.1,
  running_threshold: float = 2.0,
  thigh_length: float | None = None,
  shank_length: float | None = None,
  torso_hip_offset: float | None = None,
) -> torch.Tensor:
  """Height-aware ``variable_posture`` with leg pitch IK reference pose."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  _, joint_names = asset.find_joints(asset_cfg.joint_names)
  _, _, std_standing_vals = resolve_matching_names_values(
    data=std_standing, list_of_strings=joint_names
  )
  _, _, std_walking_vals = resolve_matching_names_values(
    data=std_walking, list_of_strings=joint_names
  )
  _, _, std_running_vals = resolve_matching_names_values(
    data=std_running, list_of_strings=joint_names
  )
  std_standing_t = torch.tensor(
    std_standing_vals, device=env.device, dtype=torch.float32
  )
  std_walking_t = torch.tensor(
    std_walking_vals, device=env.device, dtype=torch.float32
  )
  std_running_t = torch.tensor(
    std_running_vals, device=env.device, dtype=torch.float32
  )

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
  reference = _apply_height_ik_to_reference(
    env,
    asset,
    asset_cfg,
    height_command_name,
    thigh_length,
    shank_length,
    torso_hip_offset,
  )
  error_sq = torch.square(current - reference)
  return torch.exp(-torch.mean(error_sq / (std**2), dim=1))


def stand_still_height_gate(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  height_command_name: str = "base_height",
  thigh_length: float | None = None,
  shank_length: float | None = None,
  torso_hip_offset: float | None = None,
) -> torch.Tensor:
  """Standing still cost against height-IK leg/waist reference pose."""
  asset: Entity = env.scene[asset_cfg.name]
  reference = _apply_height_ik_to_reference(
    env,
    asset,
    asset_cfg,
    height_command_name,
    thigh_length,
    shank_length,
    torso_hip_offset,
  )
  diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - reference
  cost = torch.sum(torch.square(diff_angle), dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      cost *= (total_command <= command_threshold).float()
  return cost
