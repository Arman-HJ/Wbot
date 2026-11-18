"""Uniform base-height command (resampled like twist velocity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformBaseHeightCommand(CommandTerm):
  """Sample a target base height; default standing height is drawn more often."""

  cfg: UniformBaseHeightCommandCfg

  def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.height_command = torch.zeros(
      self.num_envs, 1, dtype=torch.float32, device=self.device
    )
    self.is_default_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["rel_default"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    lo, hi = self.cfg.ranges.height
    self.height_command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(lo, hi)
    self.is_default_env[env_ids] = (
      torch.empty(n, device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_default_envs
    )
    default_ids = env_ids[self.is_default_env[env_ids]]
    if len(default_ids) > 0:
      self.height_command[default_ids, 0] = self.cfg.default_height

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    self.metrics["height"][:] = self.height_command[:, 0]
    self.metrics["rel_default"][:] = self.is_default_env.float()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    del visualizer


@dataclass(kw_only=True)
class UniformBaseHeightCommandCfg(CommandTermCfg):
  """Configuration for a uniformly sampled base-height command."""

  entity_name: str
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  debug_vis: bool = False
  default_height: float = 0.95
  """Nominal standing height assigned when ``rel_default_envs`` fires."""
  rel_default_envs: float = 0.6
  """Fraction of resamples that use ``default_height`` instead of the full range."""

  @dataclass
  class Ranges:
    height: tuple[float, float] = (0.45, 1.05)

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> UniformBaseHeightCommand:
    return UniformBaseHeightCommand(self, env)
