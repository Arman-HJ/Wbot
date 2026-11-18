from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import _DEFAULT_ASSET_CFG, GaitScheduleCfg, phase, reference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def phase_observation(
  env: ManagerBasedRlEnv, schedule: GaitScheduleCfg | None = None
) -> torch.Tensor:
  p = phase(env, schedule)
  return torch.stack((torch.sin(p), torch.cos(p)), dim=-1)


def reference_joint_pos_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return (
    reference(env, asset_cfg, schedule)[0]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
