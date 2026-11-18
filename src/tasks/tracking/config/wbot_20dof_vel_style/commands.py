"""Velocity-conditioned motion command for Driving-the-Style style tracking.

Actor ``command`` is planar base velocity ``(vx, vy, wz)`` from the reference
clip (yaw frame). Full joint / body references stay available for rewards,
terminations, RSI, and the privileged critic — they are never exposed through
``command``.

Standing-start episodes (paper Sec. III-D): a fraction of resets place the
robot in the default standing pose with the reference pinned at frame 0, hold
the reference for ``standing_stab_steps``, then advance with a body-tracking
reward weight ramp (handled in ``curriculums.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityConditionedMotionCommand(MotionCommand):
  """BeyondMimic motion playback with a 3-D velocity command for the actor."""

  cfg: VelocityConditionedMotionCommandCfg

  def __init__(self, cfg: VelocityConditionedMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._is_standing_start = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._stand_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.metrics["vel_cmd_norm"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["standing_start_frac"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    """Planar reference velocity ``(vx, vy, wz)`` in the **robot** yaw/base frame.

    Paper Sec. III-B: ``v_cmd`` is the reference planar base velocity and yaw
    rate expressed in the robot base frame (the online steering command).
    """
    # Yaw-aligned robot base (drop roll/pitch) — same heading frame used at
    # gamepad deployment when the upright target only carries yaw.
    yaw_q = yaw_quat(self.robot_anchor_quat_w)
    lin_b = quat_apply_inverse(yaw_q, self.anchor_lin_vel_w)
    ang_b = quat_apply_inverse(yaw_q, self.anchor_ang_vel_w)
    return torch.cat([lin_b[:, :2], ang_b[:, 2:3]], dim=-1)

  @property
  def standing_body_weight_scale(self) -> torch.Tensor:
    """Per-env scale in ``[w_init, 1]`` for body-tracking rewards (paper ramp)."""
    n_stab = int(self.cfg.standing_stab_steps)
    n_trans = max(int(self.cfg.standing_trans_steps), 1)
    w_init = float(self.cfg.standing_body_weight_init)
    # Non-standing → 1.0. Standing: freeze at w_init during stab, then ramp.
    age = self._stand_age.float()
    phi = ((age - float(n_stab)) / float(n_trans)).clamp(0.0, 1.0)
    scale = w_init + (1.0 - w_init) * phi
    return torch.where(self._is_standing_start, scale, torch.ones_like(scale))

  def _write_standing_state(self, env_ids: torch.Tensor) -> None:
    """Place robot at default standing pose; pin reference to frame 0."""
    if env_ids.numel() == 0:
      return
    self.time_steps[env_ids] = int(self.cfg.start_time_step)
    self._is_standing_start[env_ids] = True
    self._stand_age[env_ids] = 0

    # Root from frame-0 body 0 (usually pelvis), keep xy/yaw of the clip start.
    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    # Lift / flatten: use default root z if available, else keep clip z.
    default_root = getattr(self.robot.data, "default_root_state", None)
    if default_root is not None:
      # default_root_state: (N, 13) = pos(3) quat(4) lin(3) ang(3)
      root_pos[:, 2] = default_root[env_ids, 2]

    default_q = self.robot.data.default_joint_pos[env_ids].clone()
    zero_dq = torch.zeros_like(default_q)
    self.robot.write_joint_state_to_sim(default_q, zero_dq, env_ids=env_ids)

    root_state = torch.cat(
      [
        root_pos,
        root_ori,
        torch.zeros(len(env_ids), 3, device=self.device),
        torch.zeros(len(env_ids), 3, device=self.device),
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    # Clear standing flags for the resampled set, then re-mark a subset.
    self._is_standing_start[env_ids] = False
    self._stand_age[env_ids] = 0

    # Optional force-to-clip-start (same idea as kick start_time_step_prob).
    start_prob = min(max(float(self.cfg.start_time_step_prob), 0.0), 1.0)
    if start_prob > 0.0 and env_ids.numel() > 0:
      start_mask = torch.rand(len(env_ids), device=self.device) < start_prob
      self.time_steps[env_ids[start_mask]] = int(self.cfg.start_time_step)

    p_stand = min(max(float(self.cfg.standing_start_prob), 0.0), 1.0)
    if p_stand <= 0.0 or env_ids.numel() == 0:
      self.metrics["standing_start_frac"][:] = float(self._is_standing_start.float().mean())
      self.metrics["vel_cmd_norm"][:] = torch.norm(self.command, dim=-1)
      return

    stand_mask = torch.rand(len(env_ids), device=self.device) < p_stand
    stand_ids = env_ids[stand_mask]
    self._write_standing_state(stand_ids)
    self.metrics["standing_start_frac"][:] = float(self._is_standing_start.float().mean())
    self.metrics["vel_cmd_norm"][:] = torch.norm(self.command, dim=-1)

  def _update_command(self):
    # Advance standing-start age before (possibly frozen) time-step update.
    self._stand_age = torch.where(
      self._is_standing_start,
      self._stand_age + 1,
      self._stand_age,
    )
    freeze = self._is_standing_start & (
      self._stand_age <= int(self.cfg.standing_stab_steps)
    )
    frozen_steps = self.time_steps[freeze].clone()

    super()._update_command()

    if freeze.any():
      self.time_steps[freeze] = frozen_steps

    self.metrics["vel_cmd_norm"][:] = torch.norm(self.command, dim=-1)
    self.metrics["standing_start_frac"][:] = float(
      self._is_standing_start.float().mean()
    )


@dataclass(kw_only=True)
class VelocityConditionedMotionCommandCfg(MotionCommandCfg):
  """MotionCommandCfg + velocity actor command + standing-start knobs."""

  # Present on newer mjlab; keep local defaults for older installs.
  start_time_step: int = 0
  start_time_step_prob: float = 0.0

  # Paper defaults (Sec. III-D).
  standing_start_prob: float = 0.3
  standing_stab_steps: int = 30
  standing_trans_steps: int = 60
  standing_body_weight_init: float = 0.7

  def build(self, env: ManagerBasedRlEnv) -> VelocityConditionedMotionCommand:
    return VelocityConditionedMotionCommand(self, env)
