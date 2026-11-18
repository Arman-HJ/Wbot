"""Standing-start body-tracking reward ramp (paper Sec. III-D).

Applies a soft global approximation of the per-episode weight schedule:
``w = w_base * mean_i(standing_body_weight_scale_i)``. Non-standing envs
contribute scale 1.0, so the mean stays near 1 when few standing episodes
are still in the stabilize / transition window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Body-tracking terms that paper ramps from w_init → w_final during standing-start.
_DEFAULT_BODY_TERMS: tuple[str, ...] = (
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
)


def standing_start_reward_ramp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  reward_names: Sequence[str] | None = None,
) -> dict[str, float]:
  """Scale body-tracking reward weights from the velocity-conditioned command."""
  del env_ids
  names = tuple(reward_names) if reward_names is not None else _DEFAULT_BODY_TERMS
  cmd = env.command_manager.get_term(command_name)
  scale_t = getattr(cmd, "standing_body_weight_scale", None)
  if scale_t is None:
    return {"body_weight_scale": 1.0}
  scale = float(scale_t.mean().item())

  rm = env.reward_manager
  base = getattr(env, "_vel_style_body_reward_base", None)
  if base is None:
    base = {}
    for name in names:
      if name in rm._term_names:
        idx = rm._term_names.index(name)
        base[name] = float(rm._term_cfgs[idx].weight)
    env._vel_style_body_reward_base = base  # type: ignore[attr-defined]

  for name, w0 in base.items():
    if name in rm._term_names:
      idx = rm._term_names.index(name)
      rm._term_cfgs[idx].weight = w0 * scale

  return {"body_weight_scale": scale}
