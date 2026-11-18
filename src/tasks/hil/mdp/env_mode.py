"""Per-environment HIL training mode (motion tracking vs AMP)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_HIL_MODE_ATTR = "_hil_mode"

MODE_TRACK = 0
MODE_AMP = 1


def get_hil_mode(env: ManagerBasedRlEnv) -> torch.Tensor:
  """``(num_envs,)`` int64 tensor: ``0`` = track, ``1`` = amp."""
  mode = getattr(env, _HIL_MODE_ATTR, None)
  if mode is None:
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  return mode


def init_hil_env_modes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  track_fraction: float = 0.5,
) -> None:
  """Allocate mode buffer and assign modes (startup: all envs)."""
  if not hasattr(env, _HIL_MODE_ATTR):
    setattr(
      env,
      _HIL_MODE_ATTR,
      torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
  assign_hil_env_modes(env, env_ids, track_fraction=track_fraction)


def assign_hil_env_modes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  track_fraction: float = 0.5,
) -> None:
  """Randomly assign track/amp mode for ``env_ids``."""
  mode = get_hil_mode(env)
  n = len(env_ids)
  if n == 0:
    return
  track_count = int(round(n * track_fraction))
  track_count = max(0, min(n, track_count))
  perm = torch.randperm(n, device=env.device)
  new_modes = torch.ones(n, dtype=torch.long, device=env.device)
  if track_count > 0:
    new_modes[perm[:track_count]] = MODE_TRACK
  mode[env_ids] = new_modes


def hil_track_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  mode_buf = getattr(env, _HIL_MODE_ATTR, None)
  if mode_buf is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return mode_buf == MODE_TRACK


def hil_amp_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  mode_buf = getattr(env, _HIL_MODE_ATTR, None)
  if mode_buf is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return mode_buf == MODE_AMP
