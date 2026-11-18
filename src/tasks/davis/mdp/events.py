"""Reset events for DAVIS shooting (penalty / free-kick style)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _set_root_pose_vel(
  entity,
  env_ids: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
  lin_vel: torch.Tensor | None = None,
  ang_vel: torch.Tensor | None = None,
) -> None:
  n = env_ids.numel()
  pose = torch.zeros(n, 7, device=pos.device)
  pose[:, 0:3] = pos
  pose[:, 3:7] = quat
  vel = torch.zeros(n, 6, device=pos.device)
  if lin_vel is not None:
    vel[:, 0:3] = lin_vel
  if ang_vel is not None:
    vel[:, 3:6] = ang_vel
  entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  entity.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def reset_goal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  goal_name: str = "goal",
  pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
  """Place goal at each env origin (+ optional local ``pos``) and clear score flag."""
  goal = env.scene[goal_name]
  device = env.device
  n = env_ids.numel()
  p = torch.tensor(pos, device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
  if hasattr(env.scene, "env_origins"):
    p = p + env.scene.env_origins[env_ids]
  q = torch.zeros(n, 4, device=device)
  q[:, 0] = 1.0
  if not getattr(goal, "is_fixed_base", False):
    _set_root_pose_vel(goal, env_ids, p, q)
  if hasattr(env, "_davis_scored"):
    env._davis_scored[env_ids] = 0.0  # type: ignore[attr-defined]


def _seed_score_trackers(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """After ball pose is written: clear score flag + seed cross detector (field side)."""
  ball = env.scene["ball"]
  robot = env.scene["robot"]
  n_all = env.num_envs
  if not hasattr(env, "_davis_scored"):
    env._davis_scored = torch.zeros(n_all, device=env.device)  # type: ignore
  if not hasattr(env, "_davis_ball_x_prev"):
    env._davis_ball_x_prev = ball.data.root_link_pos_w[:, 0].clone()  # type: ignore
  env._davis_scored[env_ids] = 0.0  # type: ignore[attr-defined]
  env._davis_ball_x_prev[env_ids] = ball.data.root_link_pos_w[env_ids, 0]  # type: ignore[attr-defined]
  # Seed chase progress so the first step after reset does not look like a jump.
  dist = (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1)
  if not hasattr(env, "_davis_prev_ball_dist"):
    env._davis_prev_ball_dist = dist.clone()  # type: ignore[attr-defined]
  else:
    env._davis_prev_ball_dist[env_ids] = dist[env_ids]  # type: ignore[attr-defined]


def pin_goal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  goal_name: str = "goal",
) -> None:
  """Re-nail freejoint goal to env origins every step so ball hits cannot tip it."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  reset_goal(env, env_ids, goal_name=goal_name, pos=(0.0, 0.0, 0.0))


def pin_field(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  field_name: str = "field",
) -> None:
  """Re-nail freejoint pitch (Lines + far goal) to each env origin."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if field_name not in env.scene.entities:
    return
  field = env.scene[field_name]
  device = env.device
  n = env_ids.numel()
  p = torch.zeros(n, 3, device=device)
  if hasattr(env.scene, "env_origins"):
    p = p + env.scene.env_origins[env_ids]
  q = torch.zeros(n, 4, device=device)
  q[:, 0] = 1.0
  if not getattr(field, "is_fixed_base", False):
    _set_root_pose_vel(field, env_ids, p, q)


def reset_penalty_kick(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_goal_dist: float = 1.5,
  robot_dist_range: tuple[float, float] = (2.0, 3.5),
  robot_angle_range_deg: tuple[float, float] = (-100.0, 100.0),
  ball_radius: float = 0.11,
  robot_height: float = 0.59,
) -> None:
  """Penalty-kick reset (paper App. C-A): ball 1.5 m in front of goal; robot behind ball."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  device = env.device
  n = env_ids.numel()
  origins = (
    env.scene.env_origins[env_ids]
    if hasattr(env.scene, "env_origins")
    else torch.zeros(n, 3, device=device)
  )

  # Goal at origin of each env (+x opening). Ball at +ball_goal_dist on x.
  reset_goal(env, env_ids, goal_name=goal_name, pos=(0.0, 0.0, 0.0))
  pin_field(env, env_ids)

  ball_pos = origins.clone()
  ball_pos[:, 0] += ball_goal_dist
  ball_pos[:, 2] = ball_radius
  q_id = torch.zeros(n, 4, device=device)
  q_id[:, 0] = 1.0
  _set_root_pose_vel(ball, env_ids, ball_pos, q_id)

  # Robot behind the ball (further from goal than the ball), facing the ball/goal.
  # Goal at x=0, ball at +ball_goal_dist; field opens +x. θ=0 → robot further +x.
  ang = torch.empty(n, device=device).uniform_(
    math.radians(robot_angle_range_deg[0]), math.radians(robot_angle_range_deg[1])
  )
  dist = torch.empty(n, device=device).uniform_(*robot_dist_range)
  robot_pos = ball_pos.clone()
  robot_pos[:, 0] += dist * torch.cos(ang)
  robot_pos[:, 1] += dist * torch.sin(ang)
  # Keep robot strictly field-side of the ball (never inside / behind the goal).
  robot_pos[:, 0] = torch.maximum(robot_pos[:, 0], ball_pos[:, 0] + 0.3)
  robot_pos[:, 2] = robot_height
  # Face the ball (and thus the goal).
  to_ball = ball_pos - robot_pos
  yaw = torch.atan2(to_ball[:, 1], to_ball[:, 0])
  zeros = torch.zeros_like(yaw)
  robot_quat = quat_from_euler_xyz(zeros, zeros, yaw)
  _set_root_pose_vel(robot, env_ids, robot_pos, robot_quat)

  # Joints to default.
  joint_pos = robot.data.default_joint_pos[env_ids].clone()
  joint_vel = torch.zeros_like(joint_pos)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  _seed_score_trackers(env, env_ids)


def reset_free_kick(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  robot_goal_dist: float = 5.0,
  ball_x_range: tuple[float, float] = (0.5, 4.0),
  ball_y_range: tuple[float, float] = (-3.0, 3.0),
  ball_radius: float = 0.11,
  robot_height: float = 0.59,
) -> None:
  """Free-kick reset (paper App. C-A): robot 5 m in front of goal; ball in rectangle."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  device = env.device
  n = env_ids.numel()
  origins = (
    env.scene.env_origins[env_ids]
    if hasattr(env.scene, "env_origins")
    else torch.zeros(n, 3, device=device)
  )
  reset_goal(env, env_ids, goal_name=goal_name, pos=(0.0, 0.0, 0.0))
  pin_field(env, env_ids)

  robot_pos = origins.clone()
  robot_pos[:, 0] += robot_goal_dist
  robot_pos[:, 2] = robot_height
  zeros = torch.zeros(n, device=device)
  # Face goal (−x).
  robot_quat = quat_from_euler_xyz(zeros, zeros, zeros + math.pi)
  _set_root_pose_vel(robot, env_ids, robot_pos, robot_quat)

  # Keep ball clearly in front of the goal line (x=0), never on/behind it.
  x0, x1 = ball_x_range
  x0 = max(float(x0), 0.3)
  ball_pos = origins.clone()
  ball_pos[:, 0] += torch.empty(n, device=device).uniform_(x0, float(x1))
  ball_pos[:, 1] += torch.empty(n, device=device).uniform_(*ball_y_range)
  ball_pos[:, 2] = ball_radius
  q_id = torch.zeros(n, 4, device=device)
  q_id[:, 0] = 1.0
  _set_root_pose_vel(ball, env_ids, ball_pos, q_id)

  joint_pos = robot.data.default_joint_pos[env_ids].clone()
  joint_vel = torch.zeros_like(joint_pos)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  _seed_score_trackers(env, env_ids)


def randomize_camera_extrinsics(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  sensor_name: str = "zed2i_depth",
  # Project page: translation 0.02–0.04 m, rotation 0.025–0.060 rad.
  pos_delta: float = 0.03,
  rot_delta: float = 0.04,
) -> None:
  """Startup DR on head camera pose relative to default (App. sim2real)."""
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  sensor = env.scene[sensor_name]
  cam_id = int(sensor.camera_idx)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = env_ids.numel()
  device = env.device

  # Default from sensor cfg (composed once; do not accumulate).
  default_pos = torch.tensor(sensor.cfg.pos, device=device, dtype=torch.float32)
  default_quat = torch.tensor(sensor.cfg.quat, device=device, dtype=torch.float32)

  dpos = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * pos_delta
  rpy = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * rot_delta
  dq = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
  new_quat = quat_mul(default_quat.expand(n, 4), dq)
  new_pos = default_pos.expand(n, 3) + dpos

  # Write into warp model buffers (per-env camera extrinsics).
  model = env.sim.model
  model.cam_pos[env_ids, cam_id] = new_pos.clone()
  model.cam_quat[env_ids, cam_id] = new_quat.clone()


def randomize_joint_default_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  offset_range: tuple[float, float] = (-0.02, 0.02),
) -> None:
  """Joint-position offset ±0.02 rad (project page Table sim2real)."""
  robot = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  # Offset default joint targets used by action residual.
  base = robot.data.default_joint_pos[env_ids]
  noise = torch.empty_like(base).uniform_(*offset_range)
  robot.data.default_joint_pos[env_ids] = base + noise


def reset_shoot_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  penalty_prob: float = 0.5,
  # Defaults = curriculum stage 0 (narrow). Train curriculum / play cfg may override.
  robot_dist_range: tuple[float, float] | None = None,
  robot_angle_range_deg: tuple[float, float] | None = None,
  ball_x_range: tuple[float, float] | None = None,
  ball_y_range: tuple[float, float] | None = None,
  **kwargs,
) -> None:
  """Mix penalty / free-kick resets. Curriculum can change ranges via env attrs."""
  device = env.device
  n = env_ids.numel()
  choose_penalty = torch.rand(n, device=device) < penalty_prob
  pen_ids = env_ids[choose_penalty]
  free_ids = env_ids[~choose_penalty]
  dist_range = robot_dist_range or getattr(env, "_davis_robot_dist_range", (2.0, 2.5))
  ang_range = robot_angle_range_deg or getattr(
    env, "_davis_robot_angle_range_deg", (-40.0, 40.0)
  )
  if pen_ids.numel() > 0:
    reset_penalty_kick(
      env,
      pen_ids,
      robot_dist_range=dist_range,
      robot_angle_range_deg=ang_range,
      **{k: v for k, v in kwargs.items() if k in ("ball_name", "goal_name", "asset_cfg")},
    )
  if free_ids.numel() > 0:
    bx = ball_x_range or getattr(env, "_davis_ball_x_range", (1.0, 2.0))
    by = ball_y_range or getattr(env, "_davis_ball_y_range", (-1.0, 1.0))
    reset_free_kick(
      env,
      free_ids,
      ball_x_range=bx,
      ball_y_range=by,
      **{k: v for k, v in kwargs.items() if k in ("ball_name", "goal_name", "asset_cfg")},
    )
