"""DAVIS actor / critic / aux-GT observations for wbot + 2-DoF neck."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from src.tasks.davis.mdp.depth import DepthPreprocessCfg, align_depth
from src.tasks.davis.mdp.rewards import HEAD_BODY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Scales baked into proprio (same spirit as AMP actor_frame). Deploy must match.
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05

# ZED2i FOV half-angles (project page calibrated depth ≈ 92°×60°).
HALF_HFOV_RAD = 0.80
HALF_VFOV_RAD = 0.52


def _robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def _head_pose(env: ManagerBasedRlEnv, robot: Entity, head_body: str = HEAD_BODY):
  if not hasattr(env, "_davis_head_body_id"):
    ids = robot.find_bodies(head_body, preserve_order=True)[0]
    env._davis_head_body_id = int(ids[0])  # type: ignore[attr-defined]
  bid = env._davis_head_body_id  # type: ignore[attr-defined]
  return robot.data.body_link_pos_w[:, bid], robot.data.body_link_quat_w[:, bid]


def proprio_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Single-frame proprioception: 72-D for 22 DoF (20 body + 2 neck).

  Layout (DAVIS Appendix D-A fields, dims for wbot+neck):
    base_ang_vel(3), projected_gravity(3), joint_pos_rel(22), joint_vel(22), last_action(22).
  Paper E1 shooting is 81-D (25 DoF); we recompute for platform DoF count.
  """
  robot = _robot(env, asset_cfg)
  ang = robot.data.root_link_ang_vel_b * ANG_VEL_SCALE
  grav = robot.data.projected_gravity_b
  q = robot.data.joint_pos - robot.data.default_joint_pos
  dq = robot.data.joint_vel * JOINT_VEL_SCALE
  act = env.action_manager.action
  return torch.cat([ang, grav, q, dq, act], dim=-1)


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str = "zed2i_depth",
  training: bool = True,
) -> torch.Tensor:
  """Aligned 168×80 depth (B,1,H,W). Actor-only; no ball XYZ."""
  sensor: CameraSensor = env.scene[sensor_name]
  assert sensor.data.depth is not None, f"Camera '{sensor_name}' has no depth"
  depth_m = sensor.data.depth.permute(0, 3, 1, 2).float()  # B,1,H,W
  cfg = getattr(env, "_davis_depth_cfg", None)
  if cfg is None:
    cfg = DepthPreprocessCfg()
    env._davis_depth_cfg = cfg  # type: ignore[attr-defined]
  is_train = bool(training) and not bool(getattr(env, "is_playing", False))
  return align_depth(depth_m, cfg, training=is_train)


def _head_yaw_frame_pos(
  head_pos_w: torch.Tensor,
  head_quat_w: torch.Tensor,
  point_w: torch.Tensor,
) -> torch.Tensor:
  """Object center in right-handed head-yaw-aligned frame (x fwd, y left, z up)."""
  yaw_q = yaw_quat(head_quat_w)
  return quat_apply_inverse(yaw_q, point_w - head_pos_w)


def aux_geometry_gt(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  goal_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  sensor_name: str = "zed2i_depth",
  head_body: str = HEAD_BODY,
  depth_min_m: float = 0.3,
  depth_max_m: float = 5.0,
) -> torch.Tensor:
  """Privileged aux labels: [ball_xyz, goal_xyz, ball_vis, goal_vis] in head-yaw frame."""
  del sensor_name
  from src.tasks.davis.mdp.goal import goal_pos_w

  robot = _robot(env, asset_cfg)
  ball = env.scene[ball_name]
  head_pos, head_quat = _head_pose(env, robot, head_body)
  goal_w = goal_pos_w(env, goal_name)

  ball_yaw = _head_yaw_frame_pos(head_pos, head_quat, ball.data.root_link_pos_w)
  goal_yaw = _head_yaw_frame_pos(head_pos, head_quat, goal_w)

  # Visibility with full head pose (optical ≈ head +X), FOV + depth range.
  def _visible_head(point_w: torch.Tensor) -> torch.Tensor:
    p = quat_apply_inverse(head_quat, point_w - head_pos)
    depth = p[:, 0]
    ang_y = torch.atan2(p[:, 1], p[:, 0].clamp_min(1e-4))
    ang_z = torch.atan2(p[:, 2], p[:, 0].clamp_min(1e-4))
    in_fov = (ang_y.abs() < HALF_HFOV_RAD) & (ang_z.abs() < HALF_VFOV_RAD)
    in_range = (depth >= depth_min_m) & (depth <= depth_max_m)
    return (in_fov & in_range).float().unsqueeze(-1)

  half_w = 1.20
  goal_c = goal_w
  left = goal_c.clone()
  left[:, 1] += half_w
  right = goal_c.clone()
  right[:, 1] -= half_w
  g_vis = torch.maximum(
    _visible_head(goal_c),
    torch.maximum(_visible_head(left), _visible_head(right)),
  )
  b_vis = _visible_head(ball.data.root_link_pos_w)
  return torch.cat([ball_yaw, goal_yaw, b_vis, g_vis], dim=-1)


def critic_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_name: str = "ball",
  goal_name: str = "goal",
  feet_contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Privileged critic observation (training only)."""
  robot = _robot(env, asset_cfg)
  proprio = proprio_frame(env, asset_cfg)
  lin = robot.data.root_link_lin_vel_b
  aux = aux_geometry_gt(env, ball_name, goal_name, asset_cfg)
  contact = env.scene[feet_contact_sensor_name]
  assert isinstance(contact, ContactSensor)
  if contact.data.force_history is not None:
    force = torch.linalg.vector_norm(contact.data.force_history, dim=-1).amax(dim=-1)
    feet = (force > 1.0).float()
  else:
    feet = torch.zeros(env.num_envs, 2, device=env.device)
  return torch.cat([proprio, lin, aux, feet], dim=-1)
