"""Kick-task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_kicked_success(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  success_distance: float = 3.0,
) -> torch.Tensor:
  """Terminate successfully once ball horizontal travel exceeds ``success_distance``."""
  ball = env.scene[ball_name]
  start = env._kick_ball_start_pos
  dist = torch.linalg.norm(ball.data.root_link_pos_w[:, :2] - start[:, :2], dim=-1)
  success = dist > success_distance
  env._kick_success |= success
  return success


def robot_fallen(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_height: float = 0.35,
) -> torch.Tensor:
  """Terminate when the robot pelvis drops below ``min_height``."""
  robot = env.scene[asset_cfg.name]
  return robot.data.root_link_pos_w[:, 2] < min_height
