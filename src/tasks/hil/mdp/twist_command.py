"""HIL twist command: expert playback (track) or random velocity (amp)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src.tasks.amp_v2.mdp.motion_loader import (
  _AMP_MOTION_RESET_ATTR,
  AmpMotionResetLibrary,
)
from src.tasks.hil.mdp.env_mode import MODE_AMP, MODE_TRACK, get_hil_mode
from src.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def sync_hil_twist_from_motion_frames(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  frame_indices: torch.Tensor,
  command_name: str = "twist",
) -> None:
  """Set playback cursor and track-mode twist from expert frames."""
  twist = env.command_manager.get_term(command_name)
  if hasattr(twist, "set_motion_frames"):
    twist.set_motion_frames(env_ids, frame_indices)


class HilTwistCommand(UniformVelocityCommand):
  """Twist command: track envs follow expert motion; amp envs resample randomly."""

  cfg: HilTwistCommandCfg

  def __init__(self, cfg: HilTwistCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.motion_frame_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

  def _get_library(self) -> AmpMotionResetLibrary | None:
    return getattr(self._env, _AMP_MOTION_RESET_ATTR, None)

  def set_motion_frames(
    self, env_ids: torch.Tensor, frame_indices: torch.Tensor
  ) -> None:
    self.motion_frame_idx[env_ids] = frame_indices.to(
      dtype=torch.long, device=self.device
    )
    track_ids = env_ids[get_hil_mode(self._env)[env_ids] == MODE_TRACK]
    if len(track_ids) > 0:
      self.set_command_from_frames(track_ids, self.motion_frame_idx[track_ids])

  def set_command_from_frames(
    self, env_ids: torch.Tensor, frame_indices: torch.Tensor
  ) -> None:
    library = self._get_library()
    if library is None:
      return
    lin_vel_b = library.loader.all_lin_velocities_local[frame_indices]
    ang_vel_b = library.loader.all_ang_velocities_local[frame_indices]
    self.vel_command_b[env_ids, 0] = lin_vel_b[:, 0]
    self.vel_command_b[env_ids, 1] = lin_vel_b[:, 1]
    self.vel_command_b[env_ids, 2] = ang_vel_b[:, 2]

  def resample_amp_command(self, env_ids: torch.Tensor) -> None:
    """Random body-frame twist for AMP-mode envs."""
    if len(env_ids) == 0:
      return
    self._resample_command(env_ids)
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False

  def _advance_motion_frames(self, env_ids: torch.Tensor) -> None:
    library = self._get_library()
    if library is None or len(env_ids) == 0:
      return

    next_idx = self.motion_frame_idx[env_ids] + 1
    n = library.num_frames
    if n <= 0:
      return

    wrapped = next_idx >= n
    self.motion_frame_idx[env_ids] = next_idx.remainder(n)
    if wrapped.any():
      wrap_local = wrapped.nonzero(as_tuple=False).flatten()
      wrap_global = env_ids[wrap_local]
      new_starts = library.sample_indices(len(wrap_global))
      self.motion_frame_idx[wrap_global] = new_starts

    self.set_command_from_frames(env_ids, self.motion_frame_idx[env_ids])

  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    mode = get_hil_mode(self._env)

    resample_env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
    if len(resample_env_ids) > 0:
      amp_resample = resample_env_ids[mode[resample_env_ids] == MODE_AMP]
      if len(amp_resample) > 0:
        self._resample(amp_resample)

    if dt > 0.0:
      track_ids = (mode == MODE_TRACK).nonzero(as_tuple=False).flatten()
      if len(track_ids) > 0:
        self._advance_motion_frames(track_ids)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class HilTwistCommandCfg(UniformVelocityCommandCfg):
  """HIL twist: expert playback in track mode, uniform random in amp mode."""

  heading_command: bool = False
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 0.0
  rel_forward_envs: float = 0.0

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> HilTwistCommand:
    return HilTwistCommand(self, env)

  def __post_init__(self) -> None:
    if self.heading_command:
      raise ValueError("HilTwistCommandCfg does not support heading_command=True.")
