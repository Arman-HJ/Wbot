"""WBOT 20-DoF + 2-DoF neck for DAVIS active vision.

Paper / project page: E1 has 23 body + 2 head DoFs; head yaw/pitch change the next
depth frame. Stock wbot has no neck — we graft a light 2-DoF serial chain on
``torso_link`` for DAVIS only (other tasks keep plain 20-DoF).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from src.assets.robots.wbot.wbot_20dof_constants import (
  ARMATURE_EL05,
  WBOT_20DOF_HOME,
  WBOT_ARM_DAMPING_RATIO,
  WBOT_ARM_NATURAL_FREQ,
  WBOT_HEAD_POS_IN_TORSO,
  _COLLISION_PRESETS,
  _action_scale,
  _armature_pd,
  get_spec,
)

# Head joint limits ≈ humanoid neck / E1-style active vision.
NECK_YAW_RANGE = (-1.57, 1.57)
NECK_PITCH_RANGE = (-0.90, 0.60)  # +Y axis: + looks down toward feet
NECK_JOINT_NAMES = ("neck_yaw_joint", "neck_pitch_joint")
HEAD_BODY_NAME = "head_camera_link"

# Small arm-class motor for neck (paper does not publish E1 head PD).
_NECK_ACT = _armature_pd(
  NECK_JOINT_NAMES,
  armature=ARMATURE_EL05,
  effort_limit=ElectricActuator(
    reflected_inertia=ARMATURE_EL05, velocity_limit=20.0, effort_limit=8.0
  ).effort_limit,
  w=WBOT_ARM_NATURAL_FREQ,
  zeta=WBOT_ARM_DAMPING_RATIO,
)

# MuJoCo camera on head: look along body +X (optical), +Y up → -Z look.
# Columns X=(0,-1,0), Y=(0,0,1), Z=(-1,0,0).
HEAD_CAM_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)
HEAD_CAM_POS = (0.06, 0.0, 0.04)  # slightly forward of pitch axis

# ZED2i (project page / paper): rectified depth ≈ 92° H × 60° V at 16:9-ish 168×94.
ZED2I_FOVY_DEG = 60.0
DEPTH_RENDER_W = 168
DEPTH_RENDER_H = 94


def get_spec_with_neck() -> mujoco.MjSpec:
  """20-DoF wbot + neck_yaw → neck_pitch/head_camera_link."""
  spec = get_spec()
  torso = spec.body("torso_link")

  # Remove fixed head capsule so it does not collide with the new head link.
  for g in list(torso.geoms):
    if g.name == "head_collision":
      # MjSpec has no delete_geom in older mujoco; move off and disable contact.
      g.contype = 0
      g.conaffinity = 0
      g.group = 3
      break

  yaw = torso.add_body(name="neck_yaw_link", pos=WBOT_HEAD_POS_IN_TORSO)
  yaw.add_joint(
    name="neck_yaw_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=(0, 0, 1),
    range=NECK_YAW_RANGE,
    limited=True,
  )
  yaw.add_geom(
    name="neck_yaw_visual",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(0.025,),
    rgba=(0.2, 0.2, 0.25, 1.0),
    contype=0,
    conaffinity=0,
    group=1,
  )
  yaw.mass = 0.05
  yaw.inertia = (1e-4, 1e-4, 1e-4)

  pitch = yaw.add_body(name=HEAD_BODY_NAME, pos=(0.0, 0.0, 0.04))
  pitch.add_joint(
    name="neck_pitch_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=(0, 1, 0),
    range=NECK_PITCH_RANGE,
    limited=True,
  )
  pitch.add_geom(
    name="head_camera_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.03, 0.04, 0.02),
    pos=HEAD_CAM_POS,
    rgba=(0.1, 0.55, 0.85, 1.0),
    contype=0,
    conaffinity=0,
    group=1,
  )
  pitch.mass = 0.08
  pitch.inertia = (1.5e-4, 1.5e-4, 1.5e-4)
  pitch.add_site(name="head", pos=HEAD_CAM_POS, size=(0.01,) * 3)
  return spec


def _davis_articulation(
  base: EntityArticulationInfoCfg,
  action_delay_range: tuple[int, int],
) -> EntityArticulationInfoCfg:
  art = deepcopy(base)
  neck = deepcopy(_NECK_ACT)
  delay_min, delay_max = action_delay_range
  neck.delay_min_lag = delay_min
  neck.delay_max_lag = delay_max
  neck.delay_hold_prob = 0.9 if delay_max > 0 else 0.0
  neck.delay_update_period = 4 if delay_max > 0 else 0
  # Frictionloss on soft-ankle kick plant if present on first actuator.
  if getattr(art.actuators[0], "frictionloss", 0.0):
    neck.frictionloss = art.actuators[0].frictionloss
  art.actuators = tuple(art.actuators) + (neck,)
  return art


def get_wbot_davis_robot_cfg(
  action_delay_range: tuple[int, int] = (0, 0),
  *,
  collision: Literal["full", "feet", "getup"] = "full",
  pd_gains: Literal["armature_soft_ankle", "armature_soft_ankle_pace"] = "armature_soft_ankle",
) -> EntityCfg:
  """DAVIS robot: 20 body DoF + 2 neck DoF, camera parent = head_camera_link."""
  from src.assets.robots.wbot.wbot_20dof_constants import _PD_GAIN_ARTICULATIONS

  if pd_gains not in ("armature_soft_ankle", "armature_soft_ankle_pace"):
    raise ValueError(pd_gains)
  base = _PD_GAIN_ARTICULATIONS[pd_gains]
  delay_min, delay_max = action_delay_range
  art = _davis_articulation(base, action_delay_range)
  for act in art.actuators:
    if pd_gains.endswith("_pace") and delay_max > 0:
      # Keep pace delays already set on body actuators; neck uses shared range.
      if act.target_names_expr == NECK_JOINT_NAMES or set(act.target_names_expr) <= set(
        NECK_JOINT_NAMES
      ):
        act.delay_min_lag, act.delay_max_lag = delay_min, delay_max
    else:
      act.delay_min_lag = delay_min
      act.delay_max_lag = delay_max
    act.delay_hold_prob = 0.9 if delay_max > 0 else 0.0
    act.delay_update_period = 4 if delay_max > 0 else 0

  home = deepcopy(WBOT_20DOF_HOME)
  # Mild look-down so near ball starts in FOV before policy moves the head.
  home.joint_pos = {
    **home.joint_pos,
    "neck_yaw_joint": 0.0,
    "neck_pitch_joint": 0.35,  # ≈20° down
  }

  return EntityCfg(
    init_state=home,
    collisions=(_COLLISION_PRESETS[collision],),
    spec_fn=get_spec_with_neck,
    articulation=art,
  )


def davis_action_scale(
  pd_gains: Literal["armature_soft_ankle", "armature_soft_ankle_pace"] = "armature_soft_ankle",
) -> dict[str, float]:
  from src.assets.robots.wbot.wbot_20dof_constants import _PD_GAIN_ARTICULATIONS

  art = _davis_articulation(_PD_GAIN_ARTICULATIONS[pd_gains], (0, 0))
  return _action_scale(art)
