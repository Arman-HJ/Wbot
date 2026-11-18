"""HIL observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.tasks.hil.mdp.env_mode import _HIL_MODE_ATTR

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def task_indicator(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Critic-only task mode indicator: ``0`` = track, ``1`` = amp."""
  mode_buf = getattr(env, _HIL_MODE_ATTR, None)
  if mode_buf is None:
    return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
  return mode_buf.to(dtype=torch.float32).unsqueeze(-1)
