"""WBOT 20-DOF robot asset (canonical).

Single source for MJCF, actuators, home poses, and collision presets used by
velocity / kick / tracking / AMP / suspended / getup.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import torch

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

WBOT_20DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "wbot" / "xmls" / "wbot_20dof.xml"
)
assert WBOT_20DOF_XML.exists()

# Matches ``head_collision`` on torso_link (capsule along z ≈ 0.20–0.23).
WBOT_HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.215)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  source_dir = (WBOT_20DOF_XML.parent / "meshes").resolve()
  if not source_dir.is_dir():
    raise FileNotFoundError(
      f"WBOT 20DoF meshes missing: {WBOT_20DOF_XML.parent / 'meshes'} "
      f"(resolved={source_dir})."
    )
  for file in source_dir.rglob("*"):
    if not file.is_file():
      continue
    rel_path = file.relative_to(source_dir).as_posix()
    data = file.read_bytes()
    assets[rel_path] = data
    if meshdir:
      assets[f"{meshdir}/{rel_path}"] = data
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(WBOT_20DOF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_spec_with_head() -> mujoco.MjSpec:
  """WBOT spec with a massless ``head`` site for getup rewards/terminations."""
  spec = get_spec()
  torso = spec.body("torso_link")
  if not any(s.name == "head" for s in torso.sites):
    torso.add_site(name="head", pos=WBOT_HEAD_POS_IN_TORSO)
  return spec


##
# Actuator config.
#
# Lower-body armature: ``doc/下半身转动惯量.xlsx`` column
# 「总体所有转动惯量，换算到输出端」(kg·mm²) → kg·m² (/1e6).
#   EC-A2806-P2-36   → ankle_roll
#   EC-A4310-P2-36   → hip_yaw, ankle_pitch  (36H row is nearly identical)
#   EC-A4315-P2-36   → hip_pitch, hip_roll, knee
# Upper-body armature (kg·m²):
#   shoulder pitch/roll → 0.001
#   shoulder yaw/elbow  → 0.0007
#
# PD from reflected inertia only (booster_train / armature recipe).
# The old fixed 150/100/60 kp group has been removed.
# kp = I * w^2, kd = 2 * zeta * I * w, w = 2*pi*f.
# Default (AMP / locomotion): legs f=4 Hz, ζ=1.5; arms f=10 Hz, ζ=2.
# Imitation / kick: legs f=8 Hz, ζ=1.5 (kd≤5 on hip/knee motors).
##

ARMATURE_E2806 = 5607.889 / 1e6  # 0.005607889
ARMATURE_E4310 = 24222.8413745986 / 1e6  # 0.024222841
ARMATURE_E4315 = 32980.2813208958 / 1e6  # 0.032980281
ARMATURE_RS00 = 0.001
ARMATURE_EL05 = 0.0007

ACTUATOR_RS00 = ElectricActuator(
  reflected_inertia=ARMATURE_RS00, velocity_limit=20.0, effort_limit=10.0
)
ACTUATOR_EL05 = ElectricActuator(
  reflected_inertia=ARMATURE_EL05, velocity_limit=12.0, effort_limit=4.0
)
ACTUATOR_E2806 = ElectricActuator(
  reflected_inertia=ARMATURE_E2806, velocity_limit=12.0, effort_limit=8.0
)
ACTUATOR_E4310 = ElectricActuator(
  reflected_inertia=ARMATURE_E4310, velocity_limit=12.0, effort_limit=28.0
)
ACTUATOR_E4315 = ElectricActuator(
  reflected_inertia=ARMATURE_E4315, velocity_limit=15.0, effort_limit=65.0
)

WBOT_LEG_NATURAL_FREQ = 4.0 * 2.0 * math.pi
WBOT_LEG_DAMPING_RATIO = 1.5
WBOT_ARM_NATURAL_FREQ = 10.0 * 2.0 * math.pi
WBOT_ARM_DAMPING_RATIO = 2.0
# Motion-imitation (kick / tracking): stiffer legs; kd capped by real motors (≤5).
WBOT_IMIT_LEG_NATURAL_FREQ = 8.0 * 2.0 * math.pi
WBOT_IMIT_LEG_DAMPING_RATIO = 1.5
# Kick/e1 only: leg PD at 6 Hz (AMP stays at 4 Hz). Action scale locked to
# armature (4 Hz) nominal so τ_max does not rebound with kp.
WBOT_KICK_LEG_NATURAL_FREQ = 6.0 * 2.0 * math.pi
WBOT_KICK_LEG_DAMPING_RATIO = WBOT_LEG_DAMPING_RATIO
# Back-compat aliases (old soft-ankle naming).
WBOT_SOFT_ANKLE_PITCH_NATURAL_FREQ = WBOT_KICK_LEG_NATURAL_FREQ
WBOT_SOFT_ANKLE_PITCH_DAMPING_RATIO = WBOT_KICK_LEG_DAMPING_RATIO

# Kick/e1 only (N·m). AMP / armature / imitation keep frictionloss=None.
KICK_JOINT_FRICTIONLOSS = 0.3


def _armature_pd(
  target_names_expr: tuple[str, ...],
  *,
  armature: float,
  effort_limit: float,
  w: float,
  zeta: float,
  frictionloss: float | None = None,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=armature * w**2,
    damping=2.0 * zeta * armature * w,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=frictionloss,
  )


def _with_frictionloss(
  act: BuiltinPositionActuatorCfg, frictionloss: float
) -> BuiltinPositionActuatorCfg:
  out = deepcopy(act)
  out.frictionloss = frictionloss
  return out


RS00 = _armature_pd(
  (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  armature=ARMATURE_RS00,
  effort_limit=ACTUATOR_RS00.effort_limit,
  w=WBOT_ARM_NATURAL_FREQ,
  zeta=WBOT_ARM_DAMPING_RATIO,
)
EL05 = _armature_pd(
  (".*_shoulder_yaw_joint", ".*_elbow_joint"),
  armature=ARMATURE_EL05,
  effort_limit=ACTUATOR_EL05.effort_limit,
  w=WBOT_ARM_NATURAL_FREQ,
  zeta=WBOT_ARM_DAMPING_RATIO,
)
E2806 = _armature_pd(
  (".*_ankle_roll_joint",),
  armature=ARMATURE_E2806,
  effort_limit=ACTUATOR_E2806.effort_limit,
  w=WBOT_LEG_NATURAL_FREQ,
  zeta=WBOT_LEG_DAMPING_RATIO,
)
# Split hip_yaw / ankle_pitch (kick soft-ankle can replace pitch only).
E4310_YAW = _armature_pd(
  (".*_hip_yaw_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_LEG_NATURAL_FREQ,
  zeta=WBOT_LEG_DAMPING_RATIO,
)
# AMP / default armature: same leg bandwidth as before.
E4310_ANKLE_PITCH = _armature_pd(
  (".*_ankle_pitch_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_LEG_NATURAL_FREQ,
  zeta=WBOT_LEG_DAMPING_RATIO,
)
# Kick/e1: 6 Hz legs (does not affect AMP 4 Hz).
E2806_KICK = _armature_pd(
  (".*_ankle_roll_joint",),
  armature=ARMATURE_E2806,
  effort_limit=ACTUATOR_E2806.effort_limit,
  w=WBOT_KICK_LEG_NATURAL_FREQ,
  zeta=WBOT_KICK_LEG_DAMPING_RATIO,
)
E4310_YAW_KICK = _armature_pd(
  (".*_hip_yaw_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_KICK_LEG_NATURAL_FREQ,
  zeta=WBOT_KICK_LEG_DAMPING_RATIO,
)
E4310_ANKLE_PITCH_SOFT = _armature_pd(
  (".*_ankle_pitch_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_KICK_LEG_NATURAL_FREQ,
  zeta=WBOT_KICK_LEG_DAMPING_RATIO,
)
E4315_KICK = _armature_pd(
  (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  armature=ARMATURE_E4315,
  effort_limit=ACTUATOR_E4315.effort_limit,
  w=WBOT_KICK_LEG_NATURAL_FREQ,
  zeta=WBOT_KICK_LEG_DAMPING_RATIO,
)
# Back-compat alias (hip_yaw + ankle_pitch shared group).
E4310 = _armature_pd(
  (".*_hip_yaw_joint", ".*_ankle_pitch_joint"),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_LEG_NATURAL_FREQ,
  zeta=WBOT_LEG_DAMPING_RATIO,
)
E4315 = _armature_pd(
  (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  armature=ARMATURE_E4315,
  effort_limit=ACTUATOR_E4315.effort_limit,
  w=WBOT_LEG_NATURAL_FREQ,
  zeta=WBOT_LEG_DAMPING_RATIO,
)

# Imitation legs: higher bandwidth (ankle_pitch matches other imit legs).
E2806_IMIT = _armature_pd(
  (".*_ankle_roll_joint",),
  armature=ARMATURE_E2806,
  effort_limit=ACTUATOR_E2806.effort_limit,
  w=WBOT_IMIT_LEG_NATURAL_FREQ,
  zeta=WBOT_IMIT_LEG_DAMPING_RATIO,
)
E4310_YAW_IMIT = _armature_pd(
  (".*_hip_yaw_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_IMIT_LEG_NATURAL_FREQ,
  zeta=WBOT_IMIT_LEG_DAMPING_RATIO,
)
E4310_ANKLE_PITCH_IMIT = _armature_pd(
  (".*_ankle_pitch_joint",),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_IMIT_LEG_NATURAL_FREQ,
  zeta=WBOT_IMIT_LEG_DAMPING_RATIO,
)
E4310_IMIT = _armature_pd(
  (".*_hip_yaw_joint", ".*_ankle_pitch_joint"),
  armature=ARMATURE_E4310,
  effort_limit=ACTUATOR_E4310.effort_limit,
  w=WBOT_IMIT_LEG_NATURAL_FREQ,
  zeta=WBOT_IMIT_LEG_DAMPING_RATIO,
)
E4315_IMIT = _armature_pd(
  (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  armature=ARMATURE_E4315,
  effort_limit=ACTUATOR_E4315.effort_limit,
  w=WBOT_IMIT_LEG_NATURAL_FREQ,
  zeta=WBOT_IMIT_LEG_DAMPING_RATIO,
)

# Order: arms, ankle_roll, hip_yaw+ankle_pitch, hip/knee.
# AMP walk matches booster_recipe_nodamp (shared E4310). Kick uses soft-ankle split below.
WBOT_20DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(RS00, EL05, E2806, E4310, E4315),
  soft_joint_pos_limit_factor=0.9,
)

# Kick/e1: 6 Hz legs + light dry friction (AMP untouched).
WBOT_20DOF_SOFT_ANKLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _with_frictionloss(RS00, KICK_JOINT_FRICTIONLOSS),
    _with_frictionloss(EL05, KICK_JOINT_FRICTIONLOSS),
    _with_frictionloss(E2806_KICK, KICK_JOINT_FRICTIONLOSS),
    _with_frictionloss(E4310_YAW_KICK, KICK_JOINT_FRICTIONLOSS),
    _with_frictionloss(E4310_ANKLE_PITCH_SOFT, KICK_JOINT_FRICTIONLOSS),
    _with_frictionloss(E4315_KICK, KICK_JOINT_FRICTIONLOSS),
  ),
  soft_joint_pos_limit_factor=0.9,
)

WBOT_20DOF_IMITATION_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    RS00,
    EL05,
    E2806_IMIT,
    E4310_YAW_IMIT,
    E4310_ANKLE_PITCH_IMIT,
    E4315_IMIT,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# PACE fit of the hung robot (pace-sim2real-mjlab): new collect 0.1-6 Hz ×3 equal
# envs, legs only, run 26_11_02_15-13-48, mean_140. Per-joint (armature kg·m²,
# viscous damping N·m·s/rad, Coulomb frictionloss N·m). Fitted with the
# armature-PD kp/kd below, so those gains (and the action scale) stay on the
# nominal armature. Delay still ~2.5 physics steps → WBOT_20DOF_PACE_LEG_DELAY_RANGE.
WBOT_20DOF_PACE_LEG_DYNAMICS: dict[str, tuple[float, float, float]] = {
  "left_hip_pitch_joint": (0.0189, 0.304, 0.183),
  "left_hip_roll_joint": (0.0823, 0.038, 0.295),
  "left_hip_yaw_joint": (0.0385, 0.031, 0.155),
  "left_knee_joint": (0.0695, 0.002, 0.218),
  "left_ankle_pitch_joint": (0.0269, 0.013, 0.326),
  "left_ankle_roll_joint": (0.0077, 0.001, 0.096),
  "right_hip_pitch_joint": (0.0173, 0.283, 0.206),
  "right_hip_roll_joint": (0.0788, 0.025, 0.236),
  "right_hip_yaw_joint": (0.0381, 0.021, 0.170),
  "right_knee_joint": (0.0717, 0.002, 0.260),
  "right_ankle_pitch_joint": (0.0289, 0.141, 0.268),
  "right_ankle_roll_joint": (0.0077, 0.002, 0.091),
}

# Same chirp data, arms only (run 26_10_29_13-26-12, mean_049), fitted with the
# RS00 / EL05 armature-PD gains. Arm encoder offsets reach ~0.17 rad (arms are
# not zeroed as carefully as legs) and the arm command delay is ~4.7 steps.
WBOT_20DOF_PACE_ARM_DYNAMICS: dict[str, tuple[float, float, float]] = {
  "left_shoulder_pitch_joint": (0.0156, 0.010, 0.069),
  "left_shoulder_roll_joint": (0.0125, 0.035, 0.085),
  "left_shoulder_yaw_joint": (0.0073, 0.004, 0.016),
  "left_elbow_joint": (0.0067, 0.001, 0.005),
  "right_shoulder_pitch_joint": (0.0165, 0.009, 0.062),
  "right_shoulder_roll_joint": (0.0121, 0.032, 0.078),
  "right_shoulder_yaw_joint": (0.0079, 0.004, 0.016),
  "right_elbow_joint": (0.0068, 0.001, 0.006),
}

# PACE command delay per limb, in physics steps (5 ms): legs ~2.5, arms ~4.7.
WBOT_20DOF_PACE_LEG_DELAY_RANGE = (1, 4)
WBOT_20DOF_PACE_ARM_DELAY_RANGE = (3, 7)
WBOT_20DOF_PACE_ARM_ENCODER_BIAS_RANGE = (-0.2, 0.2)


def _pace_actuators(
  act: BuiltinPositionActuatorCfg,
) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """Split one armature-PD group into per-joint actuators with PACE dynamics."""
  out = []
  dynamics = {**WBOT_20DOF_PACE_ARM_DYNAMICS, **WBOT_20DOF_PACE_LEG_DYNAMICS}
  for joint, (armature, damping, friction) in dynamics.items():
    if not any(re.fullmatch(expr, joint) for expr in act.target_names_expr):
      continue
    per_joint = deepcopy(act)
    per_joint.target_names_expr = (joint,)
    per_joint.armature = armature
    per_joint.viscous_damping = damping
    per_joint.frictionloss = friction
    out.append(per_joint)
  return tuple(out)


# Same kp/kd as WBOT_20DOF_ARTICULATION.
WBOT_20DOF_PACE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    *_pace_actuators(RS00),
    *_pace_actuators(EL05),
    *_pace_actuators(E2806),
    *_pace_actuators(E4310),
    *_pace_actuators(E4315),
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Kick 6 Hz kp/kd (same as WBOT_20DOF_SOFT_ANKLE_ARTICULATION), PACE dynamics
# (arms use the PACE frictionloss instead of KICK_JOINT_FRICTIONLOSS).
WBOT_20DOF_SOFT_ANKLE_PACE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    *_pace_actuators(RS00),
    *_pace_actuators(EL05),
    *_pace_actuators(E2806_KICK),
    *_pace_actuators(E4310_YAW_KICK),
    *_pace_actuators(E4310_ANKLE_PITCH_SOFT),
    *_pace_actuators(E4315_KICK),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def _action_scale(articulation: EntityArticulationInfoCfg) -> dict[str, float]:
  scale: dict[str, float] = {}
  for act in articulation.actuators:
    assert isinstance(act, BuiltinPositionActuatorCfg)
    assert act.effort_limit is not None
    for name in act.target_names_expr:
      scale[name] = 0.25 * float(act.effort_limit) / float(act.stiffness)
  return scale


WBOT_20DOF_ACTION_SCALE = _action_scale(WBOT_20DOF_ARTICULATION)
# Kick 6 Hz plant: scale coupled to kp (0.25·τ/stiffness), same recipe as AMP/Booster.
WBOT_20DOF_SOFT_ANKLE_ACTION_SCALE = _action_scale(WBOT_20DOF_SOFT_ANKLE_ARTICULATION)
WBOT_20DOF_IMITATION_ACTION_SCALE = _action_scale(WBOT_20DOF_IMITATION_ARTICULATION)
# Back-compat aliases.
WBOT_20DOF_ARMATURE_ARTICULATION = WBOT_20DOF_ARTICULATION
WBOT_20DOF_ARMATURE_ACTION_SCALE = WBOT_20DOF_ACTION_SCALE

##
# Keyframes.
##

# Legacy kick/tracking home (symmetric arms from gvhmr kick3 stand).
WBOT_20DOF_HOME_LEGACY = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.59),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": -0.02,
    "left_hip_roll_joint": 0.02,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    "right_ankle_roll_joint": 0.02,
    "left_ankle_roll_joint": -0.02,
    "left_shoulder_pitch_joint": -0.087,
    "left_shoulder_roll_joint": 0.297,
    "left_shoulder_yaw_joint": -0.442,
    "left_elbow_joint": 1.097,
    "right_shoulder_pitch_joint": -0.087,
    "right_shoulder_roll_joint": -0.297,
    "right_shoulder_yaw_joint": 0.442,
    "right_elbow_joint": 1.097,
  },
  joint_vel={".*": 0.0},
)
HOME_KEYFRAME = WBOT_20DOF_HOME_LEGACY

# Mean over stand.npz frames, projected onto left/right symmetry (AMP walk/run).
WBOT_20DOF_HOME_STAND = deepcopy(WBOT_20DOF_HOME_LEGACY)
WBOT_20DOF_HOME_STAND.joint_pos = {
  "left_hip_pitch_joint": -0.087935052,
  "left_hip_roll_joint": 0.042068179,
  "left_hip_yaw_joint": -0.006847813,
  "left_knee_joint": 0.174532920,
  "left_ankle_pitch_joint": -0.101345467,
  "left_ankle_roll_joint": -0.052195227,
  "right_hip_pitch_joint": -0.087935052,
  "right_hip_roll_joint": -0.042068179,
  "right_hip_yaw_joint": 0.006847813,
  "right_knee_joint": 0.174532920,
  "right_ankle_pitch_joint": -0.101345467,
  "right_ankle_roll_joint": 0.052195227,
  "left_shoulder_pitch_joint": -0.039025510,
  "left_shoulder_roll_joint": 0.231737835,
  "left_shoulder_yaw_joint": -1.003595041,
  "left_elbow_joint": 1.199648268,
  "right_shoulder_pitch_joint": -0.039025510,
  "right_shoulder_roll_joint": -0.231737835,
  "right_shoulder_yaw_joint": 1.003595041,
  "right_elbow_joint": 1.199648268,
}
WBOT_20DOF_HOME = WBOT_20DOF_HOME_STAND

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.57),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    "right_hip_roll_joint": -0.01,
    "left_hip_roll_joint": 0.01,
    ".*_knee_joint": 0.4,
    ".*_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.01,
    "left_ankle_roll_joint": -0.01,
  },
  joint_vel={".*": 0.0},
)

##
# Collision presets.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Alias used by AMP walk/run (feet contact ground; no body-ground).
WBOT_20DOF_FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Body collides with ground (lying / get-up) but not with itself.
WBOT_20DOF_GETUP_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

_COLLISION_PRESETS = {
  "full": FULL_COLLISION,
  "feet": WBOT_20DOF_FOOT_COLLISION,
  "getup": WBOT_20DOF_GETUP_COLLISION,
}

##
# Factory.
##


_PD_GAIN_ARTICULATIONS: dict[str, EntityArticulationInfoCfg] = {
  "armature": WBOT_20DOF_ARTICULATION,
  "imitation": WBOT_20DOF_IMITATION_ARTICULATION,
  "armature_soft_ankle": WBOT_20DOF_SOFT_ANKLE_ARTICULATION,
  "armature_pace": WBOT_20DOF_PACE_ARTICULATION,
  "armature_soft_ankle_pace": WBOT_20DOF_SOFT_ANKLE_PACE_ARTICULATION,
}


def actuator_group_ids(
  articulation: EntityArticulationInfoCfg, joint_exprs: tuple[str, ...]
) -> list[int]:
  """Indices into Entity actuator groups (for dr.pd_gains actuator_ids) driving any joint matching joint_exprs."""
  ids = []
  for index, act in enumerate(articulation.actuators):
    if any(re.fullmatch(j, t) for j in joint_exprs for t in act.target_names_expr):
      ids.append(index)
  return ids


@dataclass
class WbotEntityCfg(EntityCfg):
  encoder_bias_joints: tuple[str, ...] = ()
  encoder_bias_range: tuple[float, float] = (0.0, 0.0)

  def build(self) -> Entity:
    return WbotEntity(self)


class WbotEntity(Entity):
  """Samples ``encoder_bias_range`` on ``encoder_bias_joints`` once per world.

  Sampling happens at a world's first reset, i.e. after the task's startup
  events, so it overrides those columns of any task ``encoder_bias`` event.
  Later resets (episode ends, RSI teleports) keep the sampled bias.
  """

  cfg: WbotEntityCfg

  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    ids = self.find_joints(self.cfg.encoder_bias_joints, preserve_order=True)[0]
    self._bias_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
    self._bias_pending = torch.full(
      (data.nworld,), len(ids) > 0, dtype=torch.bool, device=device
    )
    self._bias_done = len(ids) == 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self._bias_done:
      return
    pending = torch.zeros_like(self._bias_pending)
    pending[slice(None) if env_ids is None else env_ids] = True
    worlds = (pending & self._bias_pending).nonzero().squeeze(-1)
    if worlds.numel() == 0:
      return
    lo, hi = self.cfg.encoder_bias_range
    bias = torch.empty(worlds.numel(), len(self._bias_joint_ids), device=worlds.device)
    self.data.encoder_bias[worlds[:, None], self._bias_joint_ids] = bias.uniform_(lo, hi)
    self._bias_pending[worlds] = False
    self._bias_done = not bool(self._bias_pending.any())


def get_wbot_20dof_robot_cfg(
  action_delay_range: tuple[int, int] = (0, 0),
  *,
  with_head_site: bool = False,
  enable_body_collision: bool = False,
  collision: Literal["full", "feet", "getup"] | None = None,
  home_pose: Literal["stand", "legacy"] = "legacy",
  pd_gains: Literal[
    "armature",
    "imitation",
    "armature_soft_ankle",
    "armature_pace",
    "armature_soft_ankle_pace",
  ] = "armature",
) -> EntityCfg:
  """Fresh WBOT 20DoF EntityCfg.

  Defaults match the historical kick/tracking stack (full collision + legacy
  home). AMP walk/run pass ``collision="feet"``, ``home_pose="stand"``; getup
  passes ``collision="getup"`` / ``enable_body_collision=True``.

  ``pd_gains``:
    - ``\"armature\"``: legs 4 Hz / ζ=1.5 (AMP / locomotion default)
    - ``\"imitation\"``: legs 8 Hz / ζ=1.5 (stiffer tracking)
    - ``\"armature_soft_ankle\"``: kick/e1 legs at 6 Hz + frictionloss (AMP stays 4 Hz)
    - ``\"armature_pace\"``: ``\"armature\"`` gains, arms and legs with PACE
      armature / damping / frictionloss per joint
    - ``\"armature_soft_ankle_pace\"``: ``\"armature_soft_ankle\"`` gains, arms
      and legs with PACE armature / damping / frictionloss per joint

  For the ``*_pace`` variants a nonzero ``action_delay_range`` (training) is
  replaced by the per-limb PACE ranges (``WBOT_20DOF_PACE_{LEG,ARM}_DELAY_RANGE``)
  and the arms get ``WBOT_20DOF_PACE_ARM_ENCODER_BIAS_RANGE`` encoder bias
  (``WbotEntity``); ``(0, 0)`` (play) disables both.
  """
  if home_pose not in ("stand", "legacy"):
    raise ValueError(f"Unknown home_pose: {home_pose}")
  if pd_gains not in _PD_GAIN_ARTICULATIONS:
    raise ValueError(
      f"Unknown pd_gains: {pd_gains} (expected one of {tuple(_PD_GAIN_ARTICULATIONS)})"
    )
  delay_min_lag, delay_max_lag = action_delay_range
  if delay_min_lag < 0 or delay_max_lag < delay_min_lag:
    raise ValueError(f"Invalid action_delay_range={action_delay_range}")

  if collision is None:
    collision = "getup" if enable_body_collision else "full"
  if collision not in _COLLISION_PRESETS:
    raise ValueError(f"Unknown collision: {collision}")

  pace = pd_gains.endswith("_pace")
  articulation = deepcopy(_PD_GAIN_ARTICULATIONS[pd_gains])
  for actuator in articulation.actuators:
    if pace and delay_max_lag > 0:
      arm = set(actuator.target_names_expr) <= set(WBOT_20DOF_PACE_ARM_DYNAMICS)
      lags = WBOT_20DOF_PACE_ARM_DELAY_RANGE if arm else WBOT_20DOF_PACE_LEG_DELAY_RANGE
      actuator.delay_min_lag, actuator.delay_max_lag = lags
    else:
      actuator.delay_min_lag = delay_min_lag
      actuator.delay_max_lag = delay_max_lag
    actuator.delay_hold_prob = 0.9 if delay_max_lag > 0 else 0.0
    actuator.delay_update_period = 4 if delay_max_lag > 0 else 0

  entity = dict(
    init_state=deepcopy(
      WBOT_20DOF_HOME if home_pose == "stand" else WBOT_20DOF_HOME_LEGACY
    ),
    collisions=(_COLLISION_PRESETS[collision],),
    spec_fn=get_spec_with_head if with_head_site else get_spec,
    articulation=articulation,
  )
  if pace and delay_max_lag > 0:
    return WbotEntityCfg(
      **entity,
      encoder_bias_joints=tuple(WBOT_20DOF_PACE_ARM_DYNAMICS),
      encoder_bias_range=WBOT_20DOF_PACE_ARM_ENCODER_BIAS_RANGE,
    )
  return EntityCfg(**entity)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wbot_20dof_robot_cfg())
  viewer.launch(robot.spec.compile())
