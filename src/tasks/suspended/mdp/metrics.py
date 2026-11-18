from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import _DEFAULT_ASSET_CFG, GaitScheduleCfg, gait_command
from .rewards import joint_pos_error, joint_vel_error

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_pos_rmse(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  return torch.mean(joint_pos_error(env, asset_cfg, schedule).square(), dim=-1).sqrt()


def joint_vel_rmse(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  return torch.mean(joint_vel_error(env, asset_cfg, schedule).square(), dim=-1).sqrt()


def gait_freq_hz(
  env: ManagerBasedRlEnv, schedule: GaitScheduleCfg | None = None
) -> torch.Tensor:
  return gait_command(env, schedule)[:, 0]


def gait_amp_scale(
  env: ManagerBasedRlEnv, schedule: GaitScheduleCfg | None = None
) -> torch.Tensor:
  return gait_command(env, schedule)[:, 1]
