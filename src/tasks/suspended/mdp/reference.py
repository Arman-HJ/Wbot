"""Sinusoidal alternating-leg (and optional arm) joint reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PERIOD = 0.8
# (first joint index of the leg in asset_cfg order, phase offset)
LEG_PHASES = ((0, 0.0), (6, math.pi))
# (joint index within the leg, amplitude in rad): hip_pitch, knee, ankle_pitch
LEG_AMPLITUDES = ((0, 0.20), (3, -0.16), (4, -0.04))
# Arms swing opposite the same-side leg (walk-like contralateral coupling).
# Applied only when these joint names are present in ``asset_cfg``.
ARM_SWINGS = (
  # (joint_name, phase_offset, amplitude_rad)
  ("left_shoulder_pitch_joint", math.pi, 0.25),
  ("right_shoulder_pitch_joint", 0.0, 0.25),
  ("left_elbow_joint", math.pi, 0.10),
  ("right_elbow_joint", 0.0, 0.10),
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass(frozen=True)
class GaitScheduleCfg:
  """Shared (all-env) gait schedule driven by the global step counter.

  One cycle sweeps ``periods`` slow→fast, holding each for ``stage_s``. Each
  cycle uses the next entry of ``amp_levels`` (blended in over ``amp_ramp_s``),
  so the full loop covers every speed × amplitude pair.
  """

  periods: tuple[float, ...] = (2.0, 1.56, 1.22, 0.95, 0.74, 0.58, 0.45, 0.35)
  stage_s: float = 3.0
  amp_levels: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
  amp_ramp_s: float = 1.0
  # Keep q_ref inside soft joint limits (knee would otherwise go below 0).
  limit_margin: float = 0.02


def _schedule_state(
  env: ManagerBasedRlEnv, cfg: GaitScheduleCfg
) -> tuple[float, float, float, float]:
  """(phase, omega, amp, amp_rate) at the current global time; same for every env."""
  t = float(env.common_step_counter) * float(env.step_dt)
  n_stage = len(cfg.periods)
  cycle_s = n_stage * cfg.stage_s
  stage_dphi = [2.0 * math.pi * cfg.stage_s / p for p in cfg.periods]
  cycle_dphi = sum(stage_dphi)

  n_cycles = int(t // cycle_s)
  tc = t - n_cycles * cycle_s
  stage = min(int(tc // cfg.stage_s), n_stage - 1)
  ts = tc - stage * cfg.stage_s
  period = cfg.periods[stage]
  omega = 2.0 * math.pi / period
  phase = n_cycles * cycle_dphi + sum(stage_dphi[:stage]) + omega * ts

  n_amp = len(cfg.amp_levels)
  a_cur = cfg.amp_levels[n_cycles % n_amp]
  a_prev = cfg.amp_levels[(n_cycles - 1) % n_amp]
  if cfg.amp_ramp_s > 0 and tc < cfg.amp_ramp_s:
    w = tc / cfg.amp_ramp_s
    amp = a_prev + (a_cur - a_prev) * w
    amp_rate = (a_cur - a_prev) / cfg.amp_ramp_s
  else:
    amp, amp_rate = a_cur, 0.0
  return phase, omega, amp, amp_rate


def phase(
  env: ManagerBasedRlEnv, schedule: GaitScheduleCfg | None = None
) -> torch.Tensor:
  if schedule is None:
    return env.episode_length_buf * env.step_dt * (2 * math.pi / PERIOD)
  p, _, _, _ = _schedule_state(env, schedule)
  return torch.full((env.num_envs,), p, device=env.device)


def gait_command(
  env: ManagerBasedRlEnv, schedule: GaitScheduleCfg | None = None
) -> torch.Tensor:
  """(gait frequency Hz, amplitude scale) per env."""
  if schedule is None:
    freq, amp = 1.0 / PERIOD, 1.0
  else:
    _, omega, amp, _ = _schedule_state(env, schedule)
    freq = omega / (2.0 * math.pi)
  out = torch.empty((env.num_envs, 2), device=env.device)
  out[:, 0] = freq
  out[:, 1] = amp
  return out


def reference(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  schedule: GaitScheduleCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference joint position and velocity in ``asset_cfg.joint_ids`` order."""
  robot = env.scene[asset_cfg.name]
  if schedule is None:
    p = phase(env)
    omega = 2 * math.pi / PERIOD
    amp, amp_rate = 1.0, 0.0
  else:
    p_f, omega, amp, amp_rate = _schedule_state(env, schedule)
    p = torch.full((env.num_envs,), p_f, device=env.device)
  q = robot.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
  dq = torch.zeros_like(q)
  for start, offset in LEG_PHASES:
    s, c = torch.sin(p + offset), torch.cos(p + offset)
    for index, amplitude in LEG_AMPLITUDES:
      q[:, start + index] += amp * amplitude * s
      dq[:, start + index] = amplitude * (amp * omega * c + amp_rate * s)

  joint_names = tuple(asset_cfg.joint_names or ())
  if joint_names:
    name_to_local = {name: i for i, name in enumerate(joint_names)}
  else:
    name_to_local = {
      name: i for i, name in enumerate(robot.joint_names) if i in asset_cfg.joint_ids
    }
  for name, offset, amplitude in ARM_SWINGS:
    local = name_to_local.get(name)
    if local is None:
      continue
    s, c = torch.sin(p + offset), torch.cos(p + offset)
    q[:, local] += amp * amplitude * s
    dq[:, local] = amplitude * (amp * omega * c + amp_rate * s)

  if schedule is not None:
    lim = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lo = lim[..., 0] + schedule.limit_margin
    hi = lim[..., 1] - schedule.limit_margin
    clipped = (q < lo) | (q > hi)
    q = torch.maximum(torch.minimum(q, hi), lo)
    dq = torch.where(clipped, torch.zeros_like(dq), dq)
  return q, dq
