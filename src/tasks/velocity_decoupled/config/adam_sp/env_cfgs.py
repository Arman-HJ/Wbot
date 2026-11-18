"""Adam SP decoupled (dual-agent) velocity tracking environment configuration."""

from __future__ import annotations

from src.assets.robots.adam_sp.adam_sp_constants import get_adam_sp_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from src.tasks.velocity_decoupled.velocity_env_cfg import (
  DecoupledRobotSpec,
  make_decoupled_velocity_env_cfg,
)

# Adam SP upper-body joints (waist + arms), 17 DoF. Legs are the remaining 12.
_ADAM_SP_UPPER_BODY_JOINT_EXPR = (
  r"waist_(roll|pitch|yaw)_joint",
  r"(left|right)_shoulder_(pitch|roll|yaw)_joint",
  r"(left|right)_elbow_joint",
  r"(left|right)_wrist_(yaw|pitch|roll)_joint",
)
_ADAM_SP_LOWER_BODY_JOINT_EXPR = (
  r"(left|right)_hip_(pitch|roll|yaw)_joint",
  r"(left|right)_knee_joint",
  r"(left|right)_ankle_(pitch|roll)_joint",
)

_ADAM_SP_SPEC = DecoupledRobotSpec(
  lower_body_joint_expr=_ADAM_SP_LOWER_BODY_JOINT_EXPR,
  upper_body_joint_expr=_ADAM_SP_UPPER_BODY_JOINT_EXPR,
  foot_sole_sites=("left_foot_sole", "right_foot_sole"),
  foot_collision_geoms=tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10)
  ),
  target_base_height=1.13,
  torso_body="torso_link",
  base_height_body="torso_link",
  imu_site="imu_in_pelvis",
  # Adam SP asset carries these sensors natively (see adam_sp.xml).
  angvel_sensor="robot/imu_ang_vel",
  linvel_sensor="robot/imu_lin_vel",
  angmom_sensor="robot/torso_angmom",
  gravity_sensor="robot/torso_upvector",
  create_torso_sensors=False,
)


def adam_sp_velocity_decoupled_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Adam SP flat-terrain decoupled dual-agent velocity task."""
  return make_decoupled_velocity_env_cfg(
    get_adam_sp_robot_cfg(),
    _ADAM_SP_SPEC,
    play=play,
    terrain="flat",
  )


def adam_sp_velocity_decoupled_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Adam SP rough-terrain decoupled dual-agent velocity task."""
  return make_decoupled_velocity_env_cfg(
    get_adam_sp_robot_cfg(),
    _ADAM_SP_SPEC,
    play=play,
    terrain="rough",
  )
