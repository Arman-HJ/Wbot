"""SMP environment helpers."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_smp_style_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Style-only SMP: zero task rewards (MimicKit single-clip / locomotion prior).

  Keep twist command present for obs layout; force standing command ranges.
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
