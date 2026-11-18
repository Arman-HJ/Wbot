"""Velocity commands for the standalone AMP locomotion task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, yaw_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv


from rsl_rl.algorithms.amp_constants import (
  COMMAND_MODE_LATERAL,
  COMMAND_MODE_MIXED,
  COMMAND_MODE_STANDING,
  COMMAND_MODE_TURNING,
)


class UniformVelocityCommand(CommandTerm):
  """Sample body-frame planar velocity and yaw-rate commands."""

  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.command_mode = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["standing_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["turning_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["lateral_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["slow_walk_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["mixed_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.is_slow_walk_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # World-frame straight reference for slow-walk rewards (set on sample).
    self.slow_walk_origin_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.slow_walk_yaw_w = torch.zeros(self.num_envs, device=self.device)
    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    episode_steps = self._env.max_episode_length
    self.metrics["error_vel_xy"] += torch.linalg.vector_norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=1
    ) / max_steps
    self.metrics["error_vel_yaw"] += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) / max_steps
    self.metrics["standing_fraction"] += (
      self.command_mode == COMMAND_MODE_STANDING
    ).float() / episode_steps
    self.metrics["turning_fraction"] += (
      self.command_mode == COMMAND_MODE_TURNING
    ).float() / episode_steps
    self.metrics["lateral_fraction"] += (
      self.command_mode == COMMAND_MODE_LATERAL
    ).float() / episode_steps
    self.metrics["slow_walk_fraction"] += self.is_slow_walk_env.float() / episode_steps
    heading_err = torch.abs(self.slow_walk_heading_error())
    self.metrics["error_heading"] += (
      heading_err * self.is_slow_walk_env.float()
    ) / max_steps
    self.metrics["mixed_fraction"] += (
      (self.command_mode == COMMAND_MODE_MIXED) & (~self.is_slow_walk_env)
    ).float() / episode_steps

  def _sample_outside_deadband(
    self,
    count: int,
    value_range: tuple[float, float],
    deadband: float,
  ) -> torch.Tensor:
    """Sample a signed command while avoiding ineffective near-zero values."""
    low, high = value_range
    if low >= -deadband or high <= deadband:
      return torch.empty(count, device=self.device).uniform_(low, high)
    negative_span = -deadband - low
    positive_span = high - deadband
    positive_probability = positive_span / (negative_span + positive_span)
    positive = torch.rand(count, device=self.device) < positive_probability
    magnitude = torch.rand(count, device=self.device)
    negative_value = low + magnitude * (-deadband - low)
    positive_value = deadband + magnitude * (high - deadband)
    return torch.where(positive, positive_value, negative_value)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    # Advanced indexing returns a copy, so ``tensor[env_ids, i].uniform_()``
    # would leave the command buffer unchanged. Sample temporary tensors and
    # assign them back explicitly.
    self.vel_command_b[env_ids, 0] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.ang_vel_z)
    draws = torch.rand(count, device=self.device)
    standing_end = self.cfg.rel_standing_envs
    turning_end = standing_end + self.cfg.rel_turning_envs
    lateral_end = turning_end + self.cfg.rel_lateral_envs
    slow_walk_end = lateral_end + self.cfg.rel_slow_walk_envs
    standing = draws < standing_end
    turning = (draws >= standing_end) & (draws < turning_end)
    lateral = (draws >= turning_end) & (draws < lateral_end)
    slow_walk = (draws >= lateral_end) & (draws < slow_walk_end)
    mixed = draws >= slow_walk_end

    self.is_standing_env[env_ids] = standing
    self.is_slow_walk_env[env_ids] = False
    self.command_mode[env_ids] = COMMAND_MODE_MIXED
    self.command_mode[env_ids[standing]] = COMMAND_MODE_STANDING
    self.command_mode[env_ids[turning]] = COMMAND_MODE_TURNING
    self.command_mode[env_ids[lateral]] = COMMAND_MODE_LATERAL
    # Slow walk stays MIXED so AMP still routes to the "walk" expert label
    # via amp_motion_velocity_threshold (vx < 0.8).
    self.command_mode[env_ids[slow_walk]] = COMMAND_MODE_MIXED
    self.command_mode[env_ids[mixed]] = COMMAND_MODE_MIXED
    self.is_slow_walk_env[env_ids[slow_walk]] = True

    self.vel_command_b[env_ids[standing]] = 0.0
    turning_ids = env_ids[turning]
    self.vel_command_b[turning_ids, :2] = 0.0
    self.vel_command_b[turning_ids, 2] = self._sample_outside_deadband(
      len(turning_ids),
      self.cfg.ranges.pure_turn_ang_vel_z,
      self.cfg.turning_deadband,
    )
    lateral_ids = env_ids[lateral]
    self.vel_command_b[lateral_ids, 0] = 0.0
    self.vel_command_b[lateral_ids, 2] = 0.0
    self.vel_command_b[lateral_ids, 1] = self._sample_outside_deadband(
      len(lateral_ids),
      self.cfg.ranges.pure_lateral_lin_vel_y,
      self.cfg.lateral_deadband,
    )
    slow_ids = env_ids[slow_walk]
    self.vel_command_b[slow_ids, 0] = torch.empty(
      len(slow_ids), device=self.device
    ).uniform_(*self.cfg.ranges.slow_walk_lin_vel_x)
    self.vel_command_b[slow_ids, 1] = 0.0
    self.vel_command_b[slow_ids, 2] = 0.0
    if len(slow_ids) > 0:
      self.slow_walk_origin_w[slow_ids] = self.robot.data.root_link_pos_w[
        slow_ids, :2
      ].clone()
      _, _, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w[slow_ids])
      self.slow_walk_yaw_w[slow_ids] = yaw

  def slow_walk_path_offset(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed lateral offset (m) and forward progress (m) vs slow-walk start line."""
    delta = self.robot.data.root_link_pos_w[:, :2] - self.slow_walk_origin_w
    cos_y = torch.cos(self.slow_walk_yaw_w)
    sin_y = torch.sin(self.slow_walk_yaw_w)
    forward = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    lateral = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    return lateral, forward

  def slow_walk_heading_error(self) -> torch.Tensor:
    """Signed yaw error (rad) vs heading frozen at slow-walk sample time."""
    _, _, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
    err = yaw - self.slow_walk_yaw_w
    return torch.atan2(torch.sin(err), torch.cos(err))

  def _update_command(self) -> None:
    self.vel_command_b[self.is_standing_env] = 0.0

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Velocity sliders for the viewed robot. Enabled by default in play."""
    del on_change, request_action
    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[0], ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[0], ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[0], ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=True)
      for label, low, high in axes:
        limit = max(abs(low), abs(high), 0.1)
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=limit,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-limit,
          max=limit,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is None or not self._joystick_enabled.value:
      return
    assert self._joystick_get_env_idx is not None
    idx = self._joystick_get_env_idx()
    for axis, slider in enumerate(self._joystick_sliders):
      self.vel_command_b[idx, axis] = slider.value
    lin = torch.linalg.vector_norm(self.vel_command_b[idx, :2])
    yaw = torch.abs(self.vel_command_b[idx, 2])
    # Keep play joystick mode boundary aligned with amp_standing_command_threshold.
    stand_thr = 0.05
    if lin < stand_thr and yaw < stand_thr:
      self.command_mode[idx] = COMMAND_MODE_STANDING
      self.is_standing_env[idx] = True
    elif lin < stand_thr:
      self.command_mode[idx] = COMMAND_MODE_TURNING
      self.is_standing_env[idx] = False
    elif yaw < stand_thr and torch.abs(self.vel_command_b[idx, 0]) < stand_thr:
      self.command_mode[idx] = COMMAND_MODE_LATERAL
      self.is_standing_env[idx] = False
    else:
      self.command_mode[idx] = COMMAND_MODE_MIXED
      self.is_standing_env[idx] = False

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the commanded planar velocity as a green world-frame arrow."""
    for env_idx in visualizer.get_env_indices(self.num_envs):
      start = self.robot.data.root_link_pos_w[env_idx].detach().cpu().numpy().copy()
      start[2] += 0.15
      command_b = torch.zeros(3, device=self.device)
      command_b[:2] = self.vel_command_b[env_idx, :2]
      command_w = quat_apply(
        yaw_quat(self.robot.data.root_link_quat_w[env_idx]), command_b
      )
      end = start + command_w.detach().cpu().numpy()
      visualizer.add_arrow(
        start=np.asarray(start),
        end=np.asarray(end),
        color=(0.1, 1.0, 0.1, 1.0),
        width=0.02,
        label="command_velocity",
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  rel_standing_envs: float = 0.0
  rel_turning_envs: float = 0.0
  rel_lateral_envs: float = 0.0
  rel_slow_walk_envs: float = 0.0
  turning_deadband: float = 0.2
  lateral_deadband: float = 0.1

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    pure_turn_ang_vel_z: tuple[float, float]
    pure_lateral_lin_vel_y: tuple[float, float]
    slow_walk_lin_vel_x: tuple[float, float] = (0.13, 0.8)

  ranges: Ranges

  def __post_init__(self) -> None:
    probabilities = (
      self.rel_standing_envs,
      self.rel_turning_envs,
      self.rel_lateral_envs,
      self.rel_slow_walk_envs,
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
      raise ValueError("Command-mode probabilities must lie in [0, 1].")
    if sum(probabilities) > 1.0:
      raise ValueError("Command-mode probabilities must sum to at most 1.")
    if self.turning_deadband < 0.0 or self.lateral_deadband < 0.0:
      raise ValueError("Command deadbands must be non-negative.")
    slow_lo, slow_hi = self.ranges.slow_walk_lin_vel_x
    if slow_hi < slow_lo:
      raise ValueError("slow_walk_lin_vel_x must satisfy low <= high.")

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)
