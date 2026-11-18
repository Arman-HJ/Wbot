"""Finite torso push, timed by simulation steps rather than wall-clock time."""
from __future__ import annotations

import math
import torch
from mjlab.utils.lab_api.math import quat_apply, yaw_quat


class ForcePulse:
  DIRECTIONS = {"向前": (1., 0., 0.), "向后": (-1., 0., 0.),
                "向左": (0., 1., 0.), "向右": (0., -1., 0.), "向下": (0., 0., -1.)}

  def __init__(self, env):
    self.env = env
    self.robot = env.scene["robot"]
    self.body_id = list(self.robot.body_names).index("torso_link")
    self.remaining = 0
    self.env_ids = None
    self.force = torch.zeros(1, 1, 3, device=env.device)
    self._label = "尚未施力"

  def start(self, direction: str, force: float, duration: float, env_idx: int = 0):
    if direction not in self.DIRECTIONS:
      raise ValueError("Unknown push direction")
    if not math.isfinite(force) or not 0 <= force <= 300:
      raise ValueError("Force must be between 0 and 300 N")
    if not math.isfinite(duration) or not 0.02 <= duration <= 1.0:
      raise ValueError("Duration must be between 0.02 and 1 second")
    if not 0 <= env_idx < self.env.num_envs:
      raise ValueError("Invalid environment index")
    self.cancel()
    self.env_ids = torch.tensor([env_idx], device=self.env.device, dtype=torch.long)
    # Lock horizontal directions to yaw at button press. Down stays world -Z.
    heading = yaw_quat(self.robot.data.root_link_quat_w[self.env_ids])
    local = torch.tensor([self.DIRECTIONS[direction]], device=self.env.device) * force
    self.force = quat_apply(heading, local).unsqueeze(1)
    self.remaining = max(1, math.ceil(duration / self.env.step_dt - 1e-9))
    self._label = f"{direction} {force:g} N / {self.remaining * self.env.step_dt:.2f} s"

  def before_step(self):
    if self.remaining:
      self.robot.write_external_wrench_to_sim(
        self.force, torch.zeros_like(self.force),
        env_ids=self.env_ids, body_ids=[self.body_id],
      )

  def after_step(self):
    if self.remaining:
      self.remaining -= 1
      if self.remaining == 0:
        self.cancel("推力已结束")

  def cancel(self, reason="推力已取消"):
    if self.env_ids is not None:
      zero = torch.zeros_like(self.force)
      self.robot.write_external_wrench_to_sim(
        zero, zero, env_ids=self.env_ids, body_ids=[self.body_id],
      )
    self.remaining = 0
    self.env_ids = None
    self._label = reason

  @property
  def status(self):
    if self.remaining:
      return f"{self._label} · 剩余 {self.remaining * self.env.step_dt:.2f} s 仿真时间"
    return self._label
