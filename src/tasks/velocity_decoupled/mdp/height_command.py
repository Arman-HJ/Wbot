"""Fixed base-height command term for decoupled velocity tracking.

Emits a constant target base height (e.g. 0.66 m) that does not change over time.
Used by the lower-body agent to regulate torso height while the upper-body agent
stabilizes posture and angular momentum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class FixedBaseHeightCommandCfg(CommandTermCfg):
  """Configuration for a fixed base-height command."""

  entity_name: str
  """Name of the entity in the scene whose height this command targets."""

  target_height: float = 0.66
  """Target root/torso height above ground (meters)."""

  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  """Height is constant; effectively never resampled (huge interval)."""

  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> "FixedBaseHeightCommand":
    return FixedBaseHeightCommand(self, env)


class FixedBaseHeightCommand(CommandTerm):
  """Constant base-height command for the lower-body agent."""

  cfg: FixedBaseHeightCommandCfg

  def __init__(self, cfg: FixedBaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.height_command = torch.full(
      (self.num_envs, 1), self.cfg.target_height, dtype=torch.float32, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Height is fixed; no-op.
    del env_ids

  def _update_command(self) -> None:
    # Constant; no-op.
    pass

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass
