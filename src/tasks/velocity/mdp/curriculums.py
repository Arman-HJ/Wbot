"""Curriculum terms for velocity / standing tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_survival(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Terrain curriculum driven by episode survival.

  - Survived to timeout → move to a harder row.
  - Failed (fall / other non-timeout termination) → move to an easier row.

  On the first ``env.reset()`` (before any ``step``), timeout/termination flags
  are not available yet, so terrain levels are left unchanged.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  if hasattr(env, "reset_time_outs") and hasattr(env, "reset_terminated"):
    move_up = env.reset_time_outs[env_ids]
    move_down = env.reset_terminated[env_ids] & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result
