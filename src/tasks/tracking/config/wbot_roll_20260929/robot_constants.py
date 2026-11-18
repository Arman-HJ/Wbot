"""WBOT_20DOF constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

WBOT_20DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "wbot" / "xmls" / "wbot_20dof.xml"
)
assert WBOT_20DOF_XML.exists()


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


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_E2806 = (0.139e-4, 0.017e-4, 0.169e-4,)
GEARS_E2806 = (1, 6, 6,)
ARMATURE_E2806 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_E2806, GEARS_E2806
)

ROTOR_INERTIAS_E4310 = (0.489e-4, 0.098e-4, 0.533e-4,)
GEARS_E4310 = (1, 6, 6,
)
ARMATURE_E4310 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_E4310, GEARS_E4310
)

ROTOR_INERTIAS_E4315 = (0.489e-4, 0.109e-4, 0.738e-4,)
GEARS_E4315 = (1, 6, 6,)
ARMATURE_E4315 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_E4315, GEARS_E4315
)

ROTOR_INERTIAS_RS00 = (0.489e-4, 0.109e-4, 0.738e-4,)
GEARS_RS00 = (1, 1, 10,)
ARMATURE_RS00 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_RS00, GEARS_RS00
)

ROTOR_INERTIAS_EL05 = (0.489e-4, 0.109e-4, 0.738e-4,)
GEARS_EL05 = (1, 1, 9,)
ARMATURE_EL05 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_EL05, GEARS_EL05
)

ACTUATOR_RS00 = ElectricActuator(
  reflected_inertia=ARMATURE_RS00,
  velocity_limit=20.0,
  effort_limit=10.0,
)
ACTUATOR_EL05 = ElectricActuator(
  reflected_inertia=ARMATURE_EL05,
  velocity_limit=12.0,
  effort_limit=4.0,
)

ACTUATOR_E2806 = ElectricActuator(
  reflected_inertia=ARMATURE_E2806,
  velocity_limit=12.0,
  effort_limit=8.0,
)
ACTUATOR_E4310 = ElectricActuator(
  reflected_inertia=ARMATURE_E4310,
  velocity_limit=12.0,
  effort_limit=28.0,
)
ACTUATOR_E4315 = ElectricActuator(
  reflected_inertia=ARMATURE_E4315,
  velocity_limit=15.0,
  effort_limit=65.0,
)

RS00 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",".*_shoulder_roll_joint"),
  stiffness=60,
  damping=3,
  effort_limit=ACTUATOR_RS00.effort_limit,
  armature=ACTUATOR_RS00.reflected_inertia,
)

EL05 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint",".*_elbow_joint"),
  stiffness=60,
  damping=3,
  effort_limit=ACTUATOR_EL05.effort_limit,
  armature=ACTUATOR_EL05.reflected_inertia,
)


E2806 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=60,
  damping=3,
  effort_limit=ACTUATOR_E2806.effort_limit,
  armature=ACTUATOR_E2806.reflected_inertia,
)

E4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",".*_ankle_pitch_joint",),
  stiffness=100,
  damping=4,
  effort_limit=ACTUATOR_E4310.effort_limit,
  armature=ACTUATOR_E4310.reflected_inertia,
)
E4315 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=150,
  damping=4,
  effort_limit=ACTUATOR_E4315.effort_limit,
  armature=ACTUATOR_E4315.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.59),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": -0.02,
    "left_hip_roll_joint": 0.02,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    "right_ankle_roll_joint": 0.02,
    "left_ankle_roll_joint": -0.02,
    # Arms: symmetric from gvhmr kick3 stand (L/R avg), elbow ~+1.1.
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
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
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

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

WBOT_20DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    RS00,
    EL05,
    E2806,
    E4310,
    E4315,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_wbot_20dof_robot_cfg() -> EntityCfg:
  """Get a fresh WBOT_20DOF robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=WBOT_20DOF_ARTICULATION,
  )


WBOT_20DOF_ACTION_SCALE: dict[str, float] = {}
for a in WBOT_20DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    WBOT_20DOF_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wbot_20dof_robot_cfg())

  viewer.launch(robot.spec.compile())
