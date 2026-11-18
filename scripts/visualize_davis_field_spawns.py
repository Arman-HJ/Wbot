#!/usr/bin/env python3
"""Top-down check: football_jax pitch + DAVIS ball/robot reset ranges.

Writes ``results/davis_field_spawns.png`` (no GPU / mjlab env needed).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assets.soccer.field import (  # noqa: E402
  FIELD_X,
  FIELD_Y,
  GOAL_BACK_OFFSET,
  GOAL_HALF_W_VIS,
  GOAL_POST_DEPTH,
)
from src.tasks.davis.mdp.curriculums import DEFAULT_STAGES  # noqa: E402

OUT = ROOT / "results" / "davis_field_spawns.png"
N_SAMPLES = 400
PENALTY_BALL_X = 1.5
RNG = np.random.default_rng(0)


def _draw_goal(ax, goal_x: float, opening_sign: float, color: str = "#f0f0f0") -> None:
  net = -opening_sign
  post_x = goal_x + net * GOAL_POST_DEPTH
  back_x = goal_x + net * GOAL_BACK_OFFSET
  for sy in (GOAL_HALF_W_VIS, -GOAL_HALF_W_VIS):
    ax.add_patch(
      Rectangle(
        (post_x - GOAL_POST_DEPTH, sy - 0.05),
        2 * GOAL_POST_DEPTH,
        0.10,
        facecolor=color,
        edgecolor="#333333",
        lw=0.6,
        zorder=3,
      )
    )
  ax.add_patch(
    Rectangle(
      (back_x - 0.05, -1.4),
      0.10,
      2.8,
      facecolor=color,
      edgecolor="#333333",
      lw=0.6,
      zorder=3,
    )
  )


def _draw_pitch(ax) -> None:
  # Outer rectangle (Lines.stl AABB shifted so goal line = 0).
  ax.add_patch(
    Rectangle(
      (0.0, -FIELD_Y),
      2.0 * FIELD_X,
      2.0 * FIELD_Y,
      fill=False,
      edgecolor="white",
      lw=2.0,
      zorder=2,
    )
  )
  # Halfway + centre circle (approx; Lines.stl has exact markings).
  ax.plot([FIELD_X, FIELD_X], [-FIELD_Y, FIELD_Y], color="white", lw=1.2, zorder=2)
  ax.add_patch(
    Circle((FIELD_X, 0.0), 1.5, fill=False, edgecolor="white", lw=1.2, zorder=2)
  )
  # Penalty boxes (approx FIFA 5v5 / fj proportions).
  for x0, sign in ((0.0, +1.0), (2.0 * FIELD_X, -1.0)):
    box_d, box_half = 3.0, 2.5
    ax.add_patch(
      Rectangle(
        (x0 if sign > 0 else x0 - box_d, -box_half),
        box_d,
        2 * box_half,
        fill=False,
        edgecolor="white",
        lw=1.0,
        zorder=2,
      )
    )
  _draw_goal(ax, 0.0, +1.0)
  _draw_goal(ax, 2.0 * FIELD_X, -1.0)
  ax.axvline(0.0, color="#ff6666", ls="--", lw=1.0, alpha=0.8, label="goal line x=0")
  ax.plot(0.0, 0.0, "r*", ms=14, zorder=5, label="scoring goal")


def _sample_penalty(n: int, dist_range, ang_range_deg) -> tuple[np.ndarray, np.ndarray]:
  ball = np.zeros((n, 2))
  ball[:, 0] = PENALTY_BALL_X
  ang = np.deg2rad(RNG.uniform(*ang_range_deg, size=n))
  dist = RNG.uniform(*dist_range, size=n)
  robot = ball.copy()
  robot[:, 0] += dist * np.cos(ang)
  robot[:, 1] += dist * np.sin(ang)
  robot[:, 0] = np.maximum(robot[:, 0], ball[:, 0] + 0.3)
  return ball, robot


def _sample_free(n: int, ball_x_range, ball_y_range, robot_goal_dist: float = 5.0):
  x0 = max(float(ball_x_range[0]), 0.3)
  ball = np.column_stack(
    [
      RNG.uniform(x0, float(ball_x_range[1]), size=n),
      RNG.uniform(*ball_y_range, size=n),
    ]
  )
  robot = np.zeros((n, 2))
  robot[:, 0] = robot_goal_dist
  return ball, robot


def main() -> None:
  OUT.parent.mkdir(parents=True, exist_ok=True)
  stages = DEFAULT_STAGES
  n_stages = len(stages)
  fig, axes = plt.subplots(1, n_stages, figsize=(6.2 * n_stages, 5.2), sharey=True)
  if n_stages == 1:
    axes = [axes]

  for ax, stage in zip(axes, stages):
    ax.set_facecolor("#1a5c2e")
    _draw_pitch(ax)

    # Half penalty / half free-kick like reset_shoot_mixed default.
    n_pen = N_SAMPLES // 2
    n_free = N_SAMPLES - n_pen
    b_p, r_p = _sample_penalty(
      n_pen, stage["robot_dist_range"], stage["robot_angle_range_deg"]
    )
    b_f, r_f = _sample_free(n_free, stage["ball_x_range"], stage["ball_y_range"])

    ax.scatter(b_p[:, 0], b_p[:, 1], s=10, c="#ff9f1c", alpha=0.55, label="ball (penalty)", zorder=4)
    ax.scatter(r_p[:, 0], r_p[:, 1], s=10, c="#4cc9f0", alpha=0.40, label="robot (penalty)", zorder=4)
    ax.scatter(
      b_f[:, 0],
      b_f[:, 1],
      s=18,
      facecolors="none",
      edgecolors="#ffd166",
      linewidths=0.8,
      alpha=0.85,
      label="ball (free-kick)",
      zorder=5,
    )
    ax.scatter(
      r_f[:, 0],
      r_f[:, 1],
      s=36,
      c="#4895ef",
      alpha=0.85,
      marker="s",
      label="robot (free-kick @ x=5)",
      zorder=5,
    )

    # One example arrow robot → ball → goal for penalty.
    ax.add_patch(
      FancyArrowPatch(
        (r_p[0, 0], r_p[0, 1]),
        (b_p[0, 0], b_p[0, 1]),
        arrowstyle="->",
        mutation_scale=10,
        color="#ffffff",
        lw=1.0,
        zorder=6,
      )
    )

    ax.set_xlim(-1.5, 2.0 * FIELD_X + 1.5)
    ax.set_ylim(-FIELD_Y - 1.0, FIELD_Y + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  ·  field opens +x toward far goal")
    ax.set_title(
      f"curriculum step≥{stage['step']}\n"
      f"penalty dist={stage['robot_dist_range']} ang={stage['robot_angle_range_deg']}°\n"
      f"free ball x={stage['ball_x_range']} y={stage['ball_y_range']}"
    )
    ax.grid(True, color="white", alpha=0.15, lw=0.5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

  axes[0].set_ylabel("y (m)")
  fig.suptitle(
    "DAVIS shoot spawns on football_jax pitch "
    f"(Lines.stl · FIELD={2*FIELD_X:.0f}×{2*FIELD_Y:.0f} m · goal @ x=0)",
    fontsize=13,
  )
  fig.tight_layout()
  fig.savefig(OUT, dpi=160)
  print(f"wrote {OUT}")
  print(
    f"pitch local: x∈[0, {2*FIELD_X}], y∈[{-FIELD_Y}, {FIELD_Y}]; "
    f"scoring goal mouth +x; far goal @ x={2*FIELD_X}"
  )


if __name__ == "__main__":
  main()
