from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_ori_6d(quat: torch.Tensor) -> torch.Tensor:
  """Flatten first two columns of rotation matrices → (N, B*6)."""
  mat = matrix_from_quat(quat)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return _body_ori_6d(ori_b)


def robot_body_pos_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Robot body/EE positions in world frame (minus env origin). Critic privileged."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idxs = _get_body_indexes(command, body_names)
  pos = command.robot_body_pos_w[:, idxs] - env.scene.env_origins[:, None, :]
  return pos.reshape(env.num_envs, -1)


def robot_body_ori_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Robot body/EE orientations in world frame (6-D per body). Critic privileged."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idxs = _get_body_indexes(command, body_names)
  return _body_ori_6d(command.robot_body_quat_w[:, idxs])


def motion_body_pos_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Motion-ref body/EE positions in world (yaw-aligned target, minus env origin)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idxs = _get_body_indexes(command, body_names)
  pos = command.body_pos_relative_w[:, idxs] - env.scene.env_origins[:, None, :]
  return pos.reshape(env.num_envs, -1)


def motion_body_ori_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Motion-ref body/EE orientations in world (yaw-aligned target, 6-D)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idxs = _get_body_indexes(command, body_names)
  return _body_ori_6d(command.body_quat_relative_w[:, idxs])


# Privileged (decoder-target) terms: per-env DR state read back from the sim.


def _model_field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
  """Per-env model field, broadcast to ``(num_envs, ...)`` if not DR-expanded."""
  value = getattr(env.sim.model, field)
  if value.shape[0] != env.num_envs:
    value = value.expand(env.num_envs, *value.shape[1:])
  return value


def robot_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Anchor body height above the env origin, (N, 1)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height.unsqueeze(-1)


def joint_encoder_bias(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Sampled joint encoder bias (rad)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.encoder_bias[:, asset_cfg.joint_ids]


def pd_gain_scale(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Position-actuator kp/kd relative to default, centered at 0: ``[kp/kp0-1, kd/kd0-1]``."""
  asset: Entity = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids
  assert ctrl_ids is not None
  kp = _model_field(env, "actuator_gainprm")[:, ctrl_ids, 0]
  kd = -_model_field(env, "actuator_biasprm")[:, ctrl_ids, 2]
  kp0 = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
  kd0 = -env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2]
  kp_scale = torch.where(kp0.abs() > 1e-6, kp / kp0.clamp_min(1e-6), torch.ones_like(kp))
  kd_scale = torch.where(kd0.abs() > 1e-6, kd / kd0.clamp_min(1e-6), torch.ones_like(kd))
  return torch.cat((kp_scale - 1.0, kd_scale - 1.0), dim=-1)


def body_mass_scale(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Body mass relative to default, centered at 0."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset.indexing.body_ids[asset_cfg.body_ids]
  mass = _model_field(env, "body_mass")[:, ids]
  mass0 = env.sim.get_default_field("body_mass")[ids]
  return mass / mass0 - 1.0


def body_com_offset(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Body COM (``body_ipos``) offset from default (m), flattened."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset.indexing.body_ids[asset_cfg.body_ids]
  ipos = _model_field(env, "body_ipos")[:, ids]
  ipos0 = env.sim.get_default_field("body_ipos")[ids]
  return (ipos - ipos0).reshape(env.num_envs, -1)


def geom_friction_coef(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Sliding friction of the first matched geom (DR uses ``shared_random``), (N, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  gid = asset.indexing.geom_ids[asset_cfg.geom_ids][:1]
  return _model_field(env, "geom_friction")[:, gid, 0]
