"""Kick-task observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_inv, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_pos_b(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  robot_name: str = "robot",
) -> torch.Tensor:
  """Ball position in robot yaw frame."""
  robot = env.scene[robot_name]
  ball = env.scene[ball_name]
  root_pos = robot.data.root_link_pos_w
  root_quat = yaw_quat(robot.data.root_link_quat_w)
  rel = ball.data.root_link_pos_w - root_pos
  return quat_apply(quat_inv(root_quat), rel)


def ball_vel_b(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  robot_name: str = "robot",
) -> torch.Tensor:
  """Ball linear velocity in robot yaw frame."""
  robot = env.scene[robot_name]
  ball = env.scene[ball_name]
  root_quat = yaw_quat(robot.data.root_link_quat_w)
  return quat_apply(quat_inv(root_quat), ball.data.root_link_lin_vel_w)
