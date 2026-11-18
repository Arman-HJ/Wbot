"""Shooting curriculum: broaden init + ramp reward groups.

论文对射门课程写了「逐渐放宽初始球位与接近难度」，并对 head/body 奖励组做 ramp
（App. C-D ablation）。具体 step 阈值未全部公开 → 做成可配置，默认见下方常量注释。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Chase-first curriculum.
# ``step`` is ``env.common_step_counter`` (≈1 per policy step; ~24 / learning iter).
# Old thresholds 8k/20k fired by ~iter 333 and *cut* approach — caused stand+look.
# ``approach_scale`` → approach_ball / adaptive_velocity / progress_to_ball
# ``contact_scale`` → kick / ball_vel / score / ankle / stagnant
DEFAULT_STAGES: list[dict[str, Any]] = [
  {
    # ~0–2000 learning iters: force chase; mute kick farming.
    "step": 0,
    "robot_dist_range": (2.0, 2.5),
    "robot_angle_range_deg": (-40.0, 40.0),
    "ball_x_range": (1.0, 2.0),
    "ball_y_range": (-1.0, 1.0),
    "align_scale": 0.5,
    "contact_scale": 0.15,
    "vision_scale": 0.4,
    "approach_scale": 2.5,
  },
  {
    # ~2000–5000 iters: keep chase strong; start enabling contact.
    "step": 48_000,
    "robot_dist_range": (2.0, 3.0),
    "robot_angle_range_deg": (-70.0, 70.0),
    "ball_x_range": (0.5, 3.0),
    "ball_y_range": (-2.0, 2.0),
    "align_scale": 0.8,
    "contact_scale": 0.5,
    "vision_scale": 0.7,
    "approach_scale": 2.0,
  },
  {
    # ~5000+ iters: shoot phase; chase still on.
    "step": 120_000,
    "robot_dist_range": (2.0, 3.5),
    "robot_angle_range_deg": (-100.0, 100.0),
    "ball_x_range": (0.5, 4.0),
    "ball_y_range": (-3.0, 3.0),
    "align_scale": 1.0,
    "contact_scale": 1.0,
    "vision_scale": 1.0,
    "approach_scale": 1.2,
  },
]


def shoot_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stages: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
  """Update reset ranges and reward weight scales from common_step_counter."""
  del env_ids
  stages = stages or DEFAULT_STAGES
  step = int(getattr(env, "common_step_counter", 0))
  # Updates ≈ env steps / rollout_length; paper anneal uses policy updates.
  # 此处用 common_step_counter 直接比较配置里的 step（配置按 env-step 写）。
  active = stages[0]
  for stage in stages:
    if step >= int(stage["step"]):
      active = stage
  env._davis_robot_dist_range = tuple(active["robot_dist_range"])  # type: ignore
  env._davis_robot_angle_range_deg = tuple(active["robot_angle_range_deg"])  # type: ignore
  env._davis_ball_x_range = tuple(active["ball_x_range"])  # type: ignore
  env._davis_ball_y_range = tuple(active["ball_y_range"])  # type: ignore
  env._davis_align_scale = float(active["align_scale"])  # type: ignore
  env._davis_contact_scale = float(active["contact_scale"])  # type: ignore
  env._davis_vision_scale = float(active["vision_scale"])  # type: ignore
  env._davis_approach_scale = float(active.get("approach_scale", 1.0))  # type: ignore

  # Scale named reward terms if present.
  rm = env.reward_manager
  vision_terms = ("head_track_ball", "keep_ball_in_horizon", "body_follow_head")
  align_terms = ("robot_behind_ball", "lateral_kick_line", "head_yaw_align")
  contact_terms = (
    "ankle_toward_ball",
    "kick_swing",
    "ball_velocity",
    "ball_velocity_direction",
    "one_shot_score",
    "stagnant_near_ball",
  )
  approach_terms = ("approach_ball", "adaptive_velocity", "progress_to_ball")
  for name in vision_terms:
    if name in rm._term_names:
      cfg = rm.get_term_cfg(name)
      base = getattr(cfg, "_davis_base_weight", cfg.weight)
      cfg._davis_base_weight = base  # type: ignore[attr-defined]
      cfg.weight = base * env._davis_vision_scale  # type: ignore[attr-defined]
  for name in align_terms:
    if name in rm._term_names:
      cfg = rm.get_term_cfg(name)
      base = getattr(cfg, "_davis_base_weight", cfg.weight)
      cfg._davis_base_weight = base  # type: ignore[attr-defined]
      cfg.weight = base * env._davis_align_scale  # type: ignore[attr-defined]
  for name in contact_terms:
    if name in rm._term_names:
      cfg = rm.get_term_cfg(name)
      base = getattr(cfg, "_davis_base_weight", cfg.weight)
      cfg._davis_base_weight = base  # type: ignore[attr-defined]
      cfg.weight = base * env._davis_contact_scale  # type: ignore[attr-defined]
  for name in approach_terms:
    if name in rm._term_names:
      cfg = rm.get_term_cfg(name)
      base = getattr(cfg, "_davis_base_weight", cfg.weight)
      cfg._davis_base_weight = base  # type: ignore[attr-defined]
      cfg.weight = base * env._davis_approach_scale  # type: ignore[attr-defined]

  return {
    "curriculum_step": float(step),
    "align_scale": float(active["align_scale"]),
    "contact_scale": float(active["contact_scale"]),
    "vision_scale": float(active["vision_scale"]),
    "approach_scale": float(active.get("approach_scale", 1.0)),
  }
