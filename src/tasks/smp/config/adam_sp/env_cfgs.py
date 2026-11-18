"""PND Adam SP SMP environment configs."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.amp_v2.config.adam_sp.env_cfgs import (
  adam_sp_amp_v2_flat_env_cfg,
  adam_sp_amp_v2_rough_env_cfg,
)


def adam_sp_smp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """SMP + amp_v2 task rewards (same env rewards / velocity commands as AMP-v2)."""
  return adam_sp_amp_v2_flat_env_cfg(play=play)


def adam_sp_smp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """SMP + amp_v2 task rewards on rough terrain."""
  return adam_sp_amp_v2_rough_env_cfg(play=play)
