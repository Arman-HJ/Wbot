"""Unitree-specific velocity observations not in mjlab.tasks.velocity.mdp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_sensor_name: str | None = "terrain_scan",
) -> torch.Tensor:
  """Body clearance above local terrain. Shape ``[B, 1]``.

  Uses the median hit height of ``terrain_sensor_name`` when present; otherwise
  returns world-frame Z (flat-plane fallback).
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    body_z = asset.data.root_link_pos_w[:, 2]

  ground_z = torch.zeros_like(body_z)
  if terrain_sensor_name is not None:
    try:
      sensor = env.scene[terrain_sensor_name]
    except KeyError:
      sensor = None
    if isinstance(sensor, RayCastSensor):
      hit_z = sensor.data.hit_pos_w[..., 2]
      valid = sensor.data.distances >= 0
      masked = torch.where(valid, hit_z, torch.full_like(hit_z, float("nan")))
      ground_z = torch.nan_to_num(torch.nanmedian(masked, dim=1).values, nan=0.0)

  return (body_z - ground_z).unsqueeze(-1)


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase_tensor = torch.zeros(env.num_envs, 2, device=env.device)
  phase_tensor[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase_tensor[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  stand_mask = (
    torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  )
  return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase_tensor), phase_tensor)


def joint_torques(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Actuator joint-space torques (``qfrc_actuator``) for selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids]


def joint_acc(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint accelerations (``qacc``) for selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_acc[:, asset_cfg.joint_ids]


def total_mass(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Current total body mass after domain randomization. Shape (N, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  masses = env.sim.model.body_mass[:, body_ids]
  return torch.sum(masses, dim=-1, keepdim=True)


def root_com_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root center-of-mass position in the root link frame. Shape (N, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  com_w = asset.data.root_com_pos_w
  root_w = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  return quat_apply_inverse(root_quat, com_w - root_w)


def joint_pd_gains(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Current joint PD gains ``[kp..., kd...]`` after domain randomization.

  Reads MuJoCo ``actuator_gainprm`` / ``actuator_biasprm`` for
  ``BuiltinPositionActuator`` (kp = gainprm[0], kd = -biasprm[2]).
  """
  asset: Entity = env.scene[asset_cfg.name]
  num_joints = asset.data.joint_pos.shape[-1]
  kp = torch.zeros(env.num_envs, num_joints, device=env.device)
  kd = torch.zeros(env.num_envs, num_joints, device=env.device)

  gainprm = env.sim.model.actuator_gainprm
  biasprm = env.sim.model.actuator_biasprm
  for actuator in asset.actuators:
    if not isinstance(actuator, BuiltinPositionActuator):
      continue
    target_ids = actuator.target_ids
    ctrl_ids = actuator.global_ctrl_ids
    kp[:, target_ids] = gainprm[:, ctrl_ids, 0]
    kd[:, target_ids] = -biasprm[:, ctrl_ids, 2]

  jnt_ids = asset_cfg.joint_ids
  return torch.cat((kp[:, jnt_ids], kd[:, jnt_ids]), dim=-1)
