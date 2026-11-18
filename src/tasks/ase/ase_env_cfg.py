"""ASE environment helpers (phase-1 pretrain / phase-2 transfer)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.ase.mdp.commands import SphericalSkillCommandCfg


def attach_skill_latent_command(
  cfg: ManagerBasedRlEnvCfg,
  *,
  latent_dim: int = 64,
  latent_steps_min: int = 1,
  latent_steps_max: int = 150,
) -> None:
  """Add spherical skill command and a dedicated ``skill`` observation group.

  The low-level policy/critic receive ``skill`` via ``obs_groups`` (concatenated
  with actor/critic). Keeping it as a separate group makes ASE diversity loss
  and phase-2 latent injection straightforward.
  """
  cfg.commands["skill"] = SphericalSkillCommandCfg(
    latent_dim=latent_dim,
    latent_steps_min=latent_steps_min,
    latent_steps_max=latent_steps_max,
  )
  cfg.observations["skill"] = ObservationGroupCfg(
    terms={
      "skill_latent": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "skill"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )


def configure_ase_pretrain_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Phase-1: no task objective — only style + MI rewards from the algorithm.

  Zero env reward weights so ``task_reward`` passed into AMP lerp is ~0.
  Keep ``twist`` command present (obs layout) but force standing / zero ranges.
  """
  for _key, term in list(cfg.rewards.items()):
    term.weight = 0.0

  twist = cfg.commands.get("twist")
  if twist is not None and hasattr(twist, "ranges"):
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(twist, "rel_standing_envs"):
      twist.rel_standing_envs = 1.0


def configure_ase_transfer_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Phase-2: keep task rewards; skill latents are driven by the high-level policy."""
  if "skill" not in cfg.commands:
    attach_skill_latent_command(cfg)
