"""ASE skill latent command: sample unit-sphere z and resample periodically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class SphericalSkillCommand(CommandTerm):
  """Latent skills z on the unit hypersphere (ASE §6.1).

  Each env keeps a latent fixed for a random number of control steps in
  ``[latent_steps_min, latent_steps_max]``, then resamples from the uniform
  spherical prior ``z = normalize(N(0, I))``.
  """

  cfg: SphericalSkillCommandCfg

  def __init__(self, cfg: SphericalSkillCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.skill = torch.zeros(self.num_envs, cfg.latent_dim, device=self.device)
    self._steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    return self.skill

  def _sample_latents(self, n: int) -> torch.Tensor:
    raw = torch.randn(n, self.cfg.latent_dim, device=self.device)
    return torch.nn.functional.normalize(raw, dim=-1)

  def _sample_horizon(self, n: int) -> torch.Tensor:
    lo = int(self.cfg.latent_steps_min)
    hi = int(self.cfg.latent_steps_max)
    if hi < lo:
      raise ValueError(f"latent_steps_max ({hi}) < latent_steps_min ({lo})")
    return torch.randint(lo, hi + 1, (n,), device=self.device, dtype=torch.long)

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.skill[env_ids] = self._sample_latents(len(env_ids))
    self._steps_left[env_ids] = self._sample_horizon(len(env_ids))

  def _update_command(self) -> None:
    self._steps_left -= 1
    expired = self._steps_left <= 0
    if torch.any(expired):
      ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._resample_command(ids)

  def set_latents(self, latents: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    """Inject latents (used by ASE phase-2 high-level policy)."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    z = torch.nn.functional.normalize(latents, dim=-1)
    self.skill[env_ids] = z
    self._steps_left[env_ids] = self._sample_horizon(len(env_ids))


@dataclass(kw_only=True)
class SphericalSkillCommandCfg(CommandTermCfg):
  """Config for :class:`SphericalSkillCommand`."""

  latent_dim: int = 64
  latent_steps_min: int = 1
  latent_steps_max: int = 150
  # Cadence is step-count based in SphericalSkillCommand; keep manager timer idle.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> SphericalSkillCommand:
    return SphericalSkillCommand(self, env)
