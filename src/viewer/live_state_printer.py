"""Periodic console print of total actuator torque during play."""

from __future__ import annotations

import time
from typing import Any

import torch


def wrap_viewer_with_live_state_print(base_cls: type, *, print_hz: float = 2.0) -> type:
  """Return a viewer subclass that prints the sum of |actuator torque|."""

  class LiveStatePrintingViewer(base_cls):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      super().__init__(*args, **kwargs)
      self._live_print_hz = max(float(print_hz), 0.1)
      self._live_print_last = 0.0

    def _live_env_idx(self) -> int:
      if hasattr(self, "env_idx"):
        return int(self.env_idx)
      scene = getattr(self, "_scene", None)
      if scene is not None and hasattr(scene, "env_idx"):
        return int(scene.env_idx)
      return 0

    def sync_env_to_viewer(self) -> None:
      super().sync_env_to_viewer()
      now = time.perf_counter()
      if now - self._live_print_last < 1.0 / self._live_print_hz:
        return
      self._live_print_last = now
      self._print_live_robot_state()

    def _print_live_robot_state(self) -> None:
      robot = self.env.unwrapped.scene["robot"]
      env_idx = self._live_env_idx()
      tau = robot.data.actuator_force[env_idx]
      tau_sum = float(torch.sum(torch.abs(tau)).item())
      step = getattr(self, "_step_count", 0)
      print(f"[LIVE] step={step} env={env_idx} tau_sum={tau_sum:.3f} Nm")

  LiveStatePrintingViewer.__name__ = f"LiveStatePrinting{base_cls.__name__}"
  LiveStatePrintingViewer.__qualname__ = LiveStatePrintingViewer.__name__
  return LiveStatePrintingViewer
