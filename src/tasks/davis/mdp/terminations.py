"""Terminations for DAVIS shooting."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def robot_fallen(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  height_threshold: float = 0.35,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return robot.data.root_link_pos_w[:, 2] < height_threshold


def bad_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  limit_angle: float = 1.0,
) -> torch.Tensor:
  """Terminate when projected gravity xy is too large (tilt beyond limit_angle)."""
  robot = env.scene[asset_cfg.name]
  grav = robot.data.projected_gravity_b
  # Upright ⇒ grav ≈ (0,0,-1); tilt magnitude ≈ ||grav_xy||.
  return torch.linalg.norm(grav[:, :2], dim=-1) > math.sin(limit_angle)


def ball_out_of_bounds(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  # Pitch (football_jax FIELD_* shifted so scoring goal line = x=0).
  field_x: float = 7.0,
  field_y: float = 4.5,
  field_margin: float = 0.1,
  # Goal-mouth region (separate from sideline / far-end). Scoring end at x=0.
  goal_half_width: float = 1.20,
  goal_depth: float = 1.00,
  goal_line_margin: float = 0.05,
  # Legacy symmetric box (kept for old configs / callers).
  limit_xy: float | None = None,
  limit_x: float | None = None,
  limit_y: float | None = None,
  # Back-compat alias for field_margin.
  margin: float | None = None,
) -> torch.Tensor:
  """Terminate when the ball leaves the *per-env local* pitch.

  DAVIS local frame: scoring goal line at x=0 (mouth opens +x), pitch
  x ∈ [0, 2·field_x], y ∈ [-field_y, field_y].

  Field body:
    - sideline: |y| > field_y + field_margin
    - past far end: x > 2·field_x + field_margin

  Goal region (independent knobs):
    - wide of posts past goal line: x < -goal_line_margin and |y| > goal_half_width
    - through / behind the net: x < -(goal_depth + goal_line_margin)
    - inside the mouth within depth is left to ``scored`` (not OOB here)

  Also: z < -0.05 (fell through ground).

  If ``limit_xy`` / ``limit_x`` / ``limit_y`` are passed (legacy), use the old
  symmetric AABB instead.
  """
  ball = env.scene[ball_name]
  p = ball.data.root_link_pos_w
  if hasattr(env.scene, "env_origins"):
    p = p - env.scene.env_origins

  if limit_xy is not None or limit_x is not None or limit_y is not None:
    lx = float(limit_x if limit_x is not None else (limit_xy or 12.0))
    ly = float(limit_y if limit_y is not None else (limit_xy or 12.0))
    return (p[:, 0].abs() > lx) | (p[:, 1].abs() > ly) | (p[:, 2] < -0.05)

  fm = float(field_margin if margin is None else margin)
  glm = float(goal_line_margin)
  half_w = float(goal_half_width)
  depth = float(goal_depth)

  sideline = p[:, 1].abs() > float(field_y) + fm
  past_far = p[:, 0] > 2.0 * float(field_x) + fm
  # Past goal line but outside the mouth (missed wide).
  wide_of_goal = (p[:, 0] < -glm) & (p[:, 1].abs() > half_w)
  # Deep behind the goal frame / through the net back.
  behind_net = p[:, 0] < -(depth + glm)
  fell = p[:, 2] < -0.05
  return sideline | past_far | wide_of_goal | behind_net | fell


def scored_success(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  goal_half_width: float = 1.20,
  goal_height: float = 1.80,
  goal_depth: float = 1.00,
) -> torch.Tensor:
  """Ball is inside the goal mouth (field is +x; interior is x <= goal_x)."""
  from src.tasks.davis.mdp.goal import goal_pos_w

  ball = env.scene[ball_name]
  p = ball.data.root_link_pos_w
  g = goal_pos_w(env, goal_name)
  in_mouth = (p[:, 1] - g[:, 1]).abs() < goal_half_width
  in_mouth = in_mouth & (p[:, 2] > 0.0) & (p[:, 2] < goal_height)
  in_net = (p[:, 0] <= g[:, 0]) & (p[:, 0] >= g[:, 0] - goal_depth)
  return in_mouth & in_net
