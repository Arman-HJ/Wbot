from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import _DEFAULT_ASSET_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def place_suspended_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Place the free-base pelvis under the hang anchor (rope taut at rest)."""
  robot = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 7:] = 0.0
  robot.write_root_state_to_sim(root, env_ids=env_ids)
