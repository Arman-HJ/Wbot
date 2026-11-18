"""Goal-directed shooting rewards (DAVIS Table XI) with active 2-DoF head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from src.tasks.davis.mdp.goal import goal_pos_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HEAD_BODY = "head_camera_link"


def _sigmoid_gate(d: torch.Tensor, d0: float, tau: float) -> torch.Tensor:
  """Paper Eq. 16: G(d; d0, τ) = σ((d0 − d)/τ)."""
  return torch.sigmoid((d0 - d) / tau)


def _ball_goal_dir(ball_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
  delta = goal_pos - ball_pos
  return delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _head_pose(env: ManagerBasedRlEnv, robot, head_body: str = HEAD_BODY):
  if not hasattr(env, "_davis_head_body_id"):
    ids = robot.find_bodies(head_body, preserve_order=True)[0]
    env._davis_head_body_id = int(ids[0])  # type: ignore[attr-defined]
  bid = env._davis_head_body_id  # type: ignore[attr-defined]
  return robot.data.body_link_pos_w[:, bid], robot.data.body_link_quat_w[:, bid]


def _yaw_angle(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Yaw (about +z) from wxyz quaternion."""
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _forward_xy_from_yaw_quat(yaw_q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = yaw_q[:, 0], yaw_q[:, 1], yaw_q[:, 2], yaw_q[:, 3]
  fx = 1 - 2 * (y * y + z * z)
  fy = 2 * (x * y + w * z)
  heading = torch.stack([fx, fy], dim=-1)
  return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def approach_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  std: float = 1.0,
  glue_dist: float = 0.45,
) -> torch.Tensor:
  """Approach ball (Table XI). Zero inside ``glue_dist`` so dribbling is not farmed.

  Without the near-ball gate, ``exp(-d)≈1`` every step while stuck on the ball
  dominates sparse ``one_shot_score``.
  """
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d = (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1)
  # Fade to 0 as the robot enters kick range (encourages transition to contact/score).
  far = ((d - glue_dist).clamp_min(0.0) / std).clamp(0.0, 8.0)
  return torch.exp(-far) * (d > glue_dist).float()


def progress_to_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense chase signal: decrease in robot–ball distance this step (meters).

  Standing yields ~0; walking closer yields positive. Seeded on reset via
  ``_seed_score_trackers``.
  """
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d = (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1)
  if not hasattr(env, "_davis_prev_ball_dist"):
    env._davis_prev_ball_dist = d.clone()  # type: ignore[attr-defined]
  prev = env._davis_prev_ball_dist  # type: ignore[attr-defined]
  # Clamp outliers (teleport / first step after reset already seeded).
  progress = (prev - d).clamp(-0.2, 0.2)
  env._davis_prev_ball_dist = d.detach().clone()  # type: ignore[attr-defined]
  return progress


def adaptive_velocity(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  target_speed: float = 0.8,
  std: float = 0.5,
  glue_dist: float = 0.45,
) -> torch.Tensor:
  """Velocity toward ball while approaching. Zero inside kick range (no dribble farm).

  ``std`` kept for API compatibility.
  """
  del std
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  to_ball = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  d = to_ball.norm(dim=-1)
  dir_xy = to_ball / d.unsqueeze(-1).clamp_min(1e-6)
  v_xy = robot.data.root_link_lin_vel_w[:, :2]
  speed_along = (v_xy * dir_xy).sum(dim=-1)
  approaching = (d > glue_dist).float()
  return approaching * torch.clamp(speed_along / target_speed, 0.0, 1.5)


def head_track_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  head_body: str = HEAD_BODY,
  std: float = 0.5,
) -> torch.Tensor:
  """Head track ball (Table XI weight 3.0): align head +X optical axis with ball."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  head_pos, head_quat = _head_pose(env, robot, head_body)
  # Body +X of head_camera_link = optical forward.
  forward = quat_apply(
    head_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  )
  to_ball = ball.data.root_link_pos_w - head_pos
  to_ball = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  cos = (forward * to_ball).sum(dim=-1).clamp(-1.0, 1.0)
  ang = torch.acos(cos)
  return torch.exp(-ang / std)


def keep_ball_in_horizon(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  head_body: str = HEAD_BODY,
  half_hfov_rad: float = 0.80,  # ZED2i ~92° H → half ≈ 0.80 rad
  half_vfov_rad: float = 0.52,  # ~60° V
) -> torch.Tensor:
  """Keep-in-horizon cost (Table XI weight −0.5) in head frame."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  head_pos, head_quat = _head_pose(env, robot, head_body)
  ball_h = quat_apply_inverse(head_quat, ball.data.root_link_pos_w - head_pos)
  ang_y = torch.atan2(ball_h[:, 1], ball_h[:, 0].clamp_min(1e-4)).abs()
  ang_z = torch.atan2(ball_h[:, 2], ball_h[:, 0].clamp_min(1e-4)).abs()
  cost_h = (ang_y / half_hfov_rad).clamp(0.0, 1.0)
  cost_v = (ang_z / half_vfov_rad).clamp(0.0, 1.0)
  return torch.maximum(cost_h, cost_v)


def body_follow_head(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  head_body: str = HEAD_BODY,
  std: float = 0.5,
) -> torch.Tensor:
  """Body follow head (Table XI weight 2.0): body yaw tracks head yaw."""
  del ball_name, goal_name
  robot = env.scene[asset_cfg.name]
  _, head_quat = _head_pose(env, robot, head_body)
  body_yaw = _yaw_angle(robot.data.root_link_quat_w)
  head_yaw = _yaw_angle(head_quat)
  ang = torch.atan2(torch.sin(head_yaw - body_yaw), torch.cos(head_yaw - body_yaw)).abs()
  return torch.exp(-ang / std)


# Back-compat aliases (old curriculum / logs).
body_track_ball = head_track_ball
body_follow_view = body_follow_head


def robot_behind_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  rho: float = 0.35,
  std: float = 0.4,
  d0: float = 1.2,
  tau: float = 0.3,
) -> torch.Tensor:
  """Behind-ball alignment (weight 1.0). Target x_behind = b − ρ d_bg (Sec. III-D1)."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  b = ball.data.root_link_pos_w
  g = goal_pos_w(env, goal_name)
  d_bg = _ball_goal_dir(b, g)
  target = b - rho * d_bg
  d = (robot.data.root_link_pos_w - target).norm(dim=-1)
  gate = _sigmoid_gate((b - robot.data.root_link_pos_w).norm(dim=-1), d0, tau)
  return gate * torch.exp(-d / std)


def lateral_kick_line_penalty(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  d0: float = 1.2,
  tau: float = 0.3,
) -> torch.Tensor:
  """Lateral deviation from ball–goal line (weight 1.0 as penalty magnitude)."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  b = ball.data.root_link_pos_w
  g = goal_pos_w(env, goal_name)
  d_bg = _ball_goal_dir(b, g)
  # Lateral unit in horizontal plane.
  lateral = torch.stack([-d_bg[:, 1], d_bg[:, 0], torch.zeros_like(d_bg[:, 0])], dim=-1)
  lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  offset = ((robot.data.root_link_pos_w - b) * lateral).sum(dim=-1).abs()
  gate = _sigmoid_gate((b - robot.data.root_link_pos_w).norm(dim=-1), d0, tau)
  return gate * offset


def head_yaw_align(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  head_body: str = HEAD_BODY,
  std: float = 0.4,
) -> torch.Tensor:
  """Head yaw align (Table XI weight 1.5): head yaw // ball→goal."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d_bg = _ball_goal_dir(ball.data.root_link_pos_w, goal_pos_w(env, goal_name))
  _, head_quat = _head_pose(env, robot, head_body)
  heading = _forward_xy_from_yaw_quat(yaw_quat(head_quat))
  dir_xy = d_bg[:, :2] / d_bg[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
  cos = (heading * dir_xy).sum(dim=-1).clamp(-1.0, 1.0)
  ang = torch.acos(cos)
  return torch.exp(-ang / std)


body_yaw_align_goal_line = head_yaw_align


def ankle_toward_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=("right_ankle_roll_link", "left_ankle_roll_link"), preserve_order=True
  ),
  d0: float = 0.8,
  tau: float = 0.2,
  std: float = 0.25,
) -> torch.Tensor:
  """Nearest ankle toward ball (weight 2.0)."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  # Resolve ankle body ids once.
  if not hasattr(env, "_davis_ankle_ids"):
    ids = robot.find_bodies(asset_cfg.body_names, preserve_order=True)[0]
    env._davis_ankle_ids = torch.tensor(ids, device=env.device, dtype=torch.long)  # type: ignore
  ankles = robot.data.body_link_pos_w[:, env._davis_ankle_ids]  # type: ignore[attr-defined]
  d = (ankles - ball.data.root_link_pos_w.unsqueeze(1)).norm(dim=-1).amin(dim=-1)
  gate = _sigmoid_gate(
    (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1), d0, tau
  )
  return gate * torch.exp(-d / std)


def ball_velocity_to_goal(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  d0: float = 0.8,
  tau: float = 0.2,
  power: float = 2.0,
  min_speed: float = 0.5,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Ball speed toward goal — **quadratic** so hard shots dominate light taps.

  Linear speed paid equally for a nudge and a drive; ``speed**power`` (default 2)
  plus ``min_speed`` floor makes a 2 m/s strike worth ~16× a 0.5 m/s tap.
  """
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d_bg = _ball_goal_dir(ball.data.root_link_pos_w, goal_pos_w(env, goal_name))
  v = ball.data.root_link_lin_vel_w
  speed_to_goal = (v * d_bg).sum(dim=-1).clamp_min(0.0)
  # Ignore weak nudges; reward only meaningful shots.
  speed_to_goal = (speed_to_goal - min_speed).clamp_min(0.0)
  gate = _sigmoid_gate(
    (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1), d0, tau
  )
  return gate * (speed_to_goal**power)


def ball_velocity_direction(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  d0: float = 0.8,
  tau: float = 0.2,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Ball velocity direction vs ball→goal (weight 1.5)."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d_bg = _ball_goal_dir(ball.data.root_link_pos_w, goal_pos_w(env, goal_name))
  v = ball.data.root_link_lin_vel_w
  speed = v.norm(dim=-1).clamp_min(1e-6)
  cos = ((v / speed.unsqueeze(-1)) * d_bg).sum(dim=-1).clamp(-1.0, 1.0)
  gate = _sigmoid_gate(
    (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1), d0, tau
  )
  # Require a real strike, not a light roll.
  moving = (speed > 0.5).float()
  return gate * moving * cos.clamp_min(0.0)


def kick_swing_to_goal(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    preserve_order=True,
  ),
  d0: float = 0.9,
  tau: float = 0.2,
  foot_std: float = 0.30,
) -> torch.Tensor:
  """Swing foot along ball→goal while near the ball (through-ball kick).

  Fixes ``touch-then-freeze``: proximity rewards saturate on a light tap; this
  pays for ankle speed in the shot direction when the foot is close to the ball.
  """
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  if not hasattr(env, "_davis_ankle_ids"):
    ids = robot.find_bodies(asset_cfg.body_names, preserve_order=True)[0]
    env._davis_ankle_ids = torch.tensor(ids, device=env.device, dtype=torch.long)  # type: ignore
  aids = env._davis_ankle_ids  # type: ignore[attr-defined]
  ankles_p = robot.data.body_link_pos_w[:, aids]
  ankles_v = robot.data.body_link_lin_vel_w[:, aids]
  ball_p = ball.data.root_link_pos_w
  d_bg = _ball_goal_dir(ball_p, goal_pos_w(env, goal_name)).unsqueeze(1)  # B,1,3
  # Speed of each ankle along shot direction; take the faster foot.
  swing = (ankles_v * d_bg).sum(dim=-1).clamp_min(0.0).amax(dim=-1)
  foot_dist = (ankles_p - ball_p.unsqueeze(1)).norm(dim=-1).amin(dim=-1)
  foot_near = torch.exp(-foot_dist / foot_std)
  gate = _sigmoid_gate((ball_p - robot.data.root_link_pos_w).norm(dim=-1), d0, tau)
  return gate * foot_near * swing


def stagnant_near_ball(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  near_dist: float = 0.55,
  ball_speed_max: float = 0.4,
) -> torch.Tensor:
  """Penalty: robot loitering next to a nearly-still ball (touch-then-stop)."""
  robot = env.scene[asset_cfg.name]
  ball = env.scene[ball_name]
  d = (ball.data.root_link_pos_w - robot.data.root_link_pos_w).norm(dim=-1)
  ball_speed = ball.data.root_link_lin_vel_w.norm(dim=-1)
  near = (d < near_dist).float()
  slow = (ball_speed < ball_speed_max).float()
  return near * slow


def one_shot_score(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  goal_half_width: float = 1.20,
  goal_height: float = 1.80,
  goal_depth: float = 1.00,
) -> torch.Tensor:
  """Sparse score when the ball crosses the goal line from the field once.

  Scene: goal at env origin, mouth opens toward +x (field is x > goal_x). A goal
  requires the ball to move from in front of the line (x > goal_x) to on/behind
  it (goal_x - depth <= x <= goal_x), inside the mouth. Birth poses with x > goal_x
  must NOT count as scored.
  """
  ball = env.scene[ball_name]
  p = ball.data.root_link_pos_w
  g = goal_pos_w(env, goal_name)
  if not hasattr(env, "_davis_scored"):
    env._davis_scored = torch.zeros(env.num_envs, device=env.device)  # type: ignore
  if not hasattr(env, "_davis_ball_x_prev"):
    env._davis_ball_x_prev = p[:, 0].clone()  # type: ignore

  prev_x = env._davis_ball_x_prev  # type: ignore[attr-defined]
  in_mouth = (p[:, 1] - g[:, 1]).abs() < goal_half_width
  in_mouth = in_mouth & (p[:, 2] > 0.0) & (p[:, 2] < goal_height)
  behind_or_on_line = (p[:, 0] <= g[:, 0]) & (p[:, 0] >= g[:, 0] - goal_depth)
  was_in_front = prev_x > g[:, 0]
  crossed = was_in_front & behind_or_on_line & in_mouth
  scored = crossed.float()
  already = env._davis_scored  # type: ignore[attr-defined]
  fresh = scored * (1.0 - already)
  env._davis_scored = torch.maximum(already, scored)  # type: ignore[attr-defined]
  env._davis_ball_x_prev = p[:, 0].clone()  # type: ignore[attr-defined]
  return fresh


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def joint_limits_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  soft = robot.data.soft_joint_pos_limits
  q = robot.data.joint_pos
  lo = (soft[:, :, 0] - q).clamp_min(0.0)
  hi = (q - soft[:, :, 1]).clamp_min(0.0)
  return torch.sum(lo * lo + hi * hi, dim=-1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  grav = robot.data.projected_gravity_b
  return torch.sum(torch.square(grav[:, :2]), dim=-1)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])
