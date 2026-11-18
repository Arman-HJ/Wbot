"""Goal pose helpers for multi-env DAVIS shooting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def goal_pos_w(env: ManagerBasedRlEnv, goal_name: str = "goal") -> torch.Tensor:
  """World-frame goal origin for every env.

  Fixed-base goals report ``root_link_pos_w = 0`` in mjlab even when each Warp
  world is offset; fall back to ``env_origins`` so rewards/obs stay local.
  Freejoint goals that we pin to the env origin use the entity root pose.
  """
  goal = env.scene[goal_name]
  if getattr(goal, "is_fixed_base", False) and hasattr(env.scene, "env_origins"):
    return env.scene.env_origins
  return goal.data.root_link_pos_w
