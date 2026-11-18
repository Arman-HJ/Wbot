from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import _DEFAULT_ASSET_CFG, GaitScheduleCfg, reference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_pos_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return robot.data.joint_pos[:, asset_cfg.joint_ids] - reference(env, asset_cfg, schedule)[0]


def joint_vel_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return robot.data.joint_vel[:, asset_cfg.joint_ids] - reference(env, asset_cfg, schedule)[1]


def track_joint_pos(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  error = joint_pos_error(env, asset_cfg, schedule)
  return torch.exp(-torch.mean(error.square(), dim=-1) / std**2)


def track_joint_vel(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  error = joint_vel_error(env, asset_cfg, schedule)
  return torch.exp(-torch.mean(error.square(), dim=-1) / std**2)


def base_ang_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Squared pelvis angular velocity (body frame); penalizes swinging on the hang ropes."""
  robot = env.scene[asset_cfg.name]
  return torch.sum(robot.data.root_link_ang_vel_b.square(), dim=1)


def energy_cost(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Approximate mechanical power: sum of |torque * velocity|."""
  robot = env.scene[asset_cfg.name]
  joint_torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_torques * joint_vel), dim=1)
