"""AgiBot X2 decoupled (dual-agent) velocity tracking environment configuration."""

from __future__ import annotations

from src.assets.robots import AGIBOT_X2_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
from src.tasks.velocity_decoupled.velocity_env_cfg import (
  DecoupledRobotSpec,
  make_decoupled_velocity_env_cfg,
)

# X2 upper-body joints (torso + arms), 17 DoF. Legs are the remaining 12.
_X2_UPPER_BODY_JOINT_EXPR = (
  r"waist_(yaw|pitch|roll)_joint",
  r"(left|right)_shoulder_(pitch|roll|yaw)_joint",
  r"(left|right)_elbow_joint",
  r"(left|right)_wrist_(yaw|pitch|roll)_joint",
)
_X2_LOWER_BODY_JOINT_EXPR = (
  r"(left|right)_hip_(pitch|roll|yaw)_joint",
  r"(left|right)_knee_joint",
  r"(left|right)_ankle_(pitch|roll)_joint",
)

_X2_SPEC = DecoupledRobotSpec(
  lower_body_joint_expr=_X2_LOWER_BODY_JOINT_EXPR,
  upper_body_joint_expr=_X2_UPPER_BODY_JOINT_EXPR,
  foot_sole_sites=("left_foot", "right_foot"),
  foot_collision_geoms=tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  ),
  target_base_height=0.66,
  torso_body="torso_link",
  base_height_body="torso_link",
  angvel_sensor="robot/imu_torso_ang_vel",
  linvel_sensor="robot/imu_torso_lin_vel",
  angmom_sensor="robot/torso_angmom",
  gravity_sensor="robot/imu_upvector",
)


def agibot_x2_velocity_decoupled_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """AgiBot X2 flat-terrain decoupled dual-agent velocity task."""
  return make_decoupled_velocity_env_cfg(
    AGIBOT_X2_CFG,
    _X2_SPEC,
    play=play,
    terrain="flat",
  )


def agibot_x2_velocity_decoupled_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """AgiBot X2 rough-terrain decoupled dual-agent velocity task."""
  return make_decoupled_velocity_env_cfg(
    AGIBOT_X2_CFG,
    _X2_SPEC,
    play=play,
    terrain="rough",
  )
