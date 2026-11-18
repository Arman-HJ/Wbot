"""PND Adam SP ASE environment configs."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.amp_v2.config.adam_sp.env_cfgs import (
  adam_sp_amp_v2_flat_env_cfg,
  adam_sp_amp_v2_rough_env_cfg,
)
from src.tasks.ase.ase_env_cfg import (
  attach_skill_latent_command,
  configure_ase_pretrain_env,
  configure_ase_transfer_env,
)
from src.tasks.ase.config.adam_sp.rl_cfg import (
  _ASE_SMALL_LATENT_DIM,
  _ASE_SMALL_LATENT_STEPS,
)


def adam_sp_ase_rough_env_cfg(
  play: bool = False, *, latent_dim: int = _ASE_SMALL_LATENT_DIM
) -> ManagerBasedRlEnvCfg:
  """Phase-1 ASE pretrain on rough terrain (style + MI only)."""
  cfg = adam_sp_amp_v2_rough_env_cfg(play=play)
  attach_skill_latent_command(
    cfg,
    latent_dim=latent_dim,
    latent_steps_min=_ASE_SMALL_LATENT_STEPS[0],
    latent_steps_max=_ASE_SMALL_LATENT_STEPS[1],
  )
  if not play:
    configure_ase_pretrain_env(cfg)
  return cfg


def adam_sp_ase_flat_env_cfg(
  play: bool = False, *, latent_dim: int = _ASE_SMALL_LATENT_DIM
) -> ManagerBasedRlEnvCfg:
  """Phase-1 ASE pretrain on flat terrain."""
  cfg = adam_sp_amp_v2_flat_env_cfg(play=play)
  attach_skill_latent_command(
    cfg,
    latent_dim=latent_dim,
    latent_steps_min=_ASE_SMALL_LATENT_STEPS[0],
    latent_steps_max=_ASE_SMALL_LATENT_STEPS[1],
  )
  if not play:
    configure_ase_pretrain_env(cfg)
  return cfg


def adam_sp_ase_transfer_flat_env_cfg(
  play: bool = False, *, latent_dim: int = _ASE_SMALL_LATENT_DIM
) -> ManagerBasedRlEnvCfg:
  """Phase-2 velocity transfer on flat terrain (task rewards kept)."""
  cfg = adam_sp_amp_v2_flat_env_cfg(play=play)
  attach_skill_latent_command(
    cfg,
    latent_dim=latent_dim,
    latent_steps_min=1,
    latent_steps_max=150,
  )
  configure_ase_transfer_env(cfg)
  return cfg


def adam_sp_ase_transfer_rough_env_cfg(
  play: bool = False, *, latent_dim: int = _ASE_SMALL_LATENT_DIM
) -> ManagerBasedRlEnvCfg:
  """Phase-2 velocity transfer on rough terrain."""
  cfg = adam_sp_amp_v2_rough_env_cfg(play=play)
  attach_skill_latent_command(
    cfg,
    latent_dim=latent_dim,
    latent_steps_min=1,
    latent_steps_max=150,
  )
  configure_ase_transfer_env(cfg)
  return cfg
