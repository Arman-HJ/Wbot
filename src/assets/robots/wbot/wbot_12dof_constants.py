"""WBOT_12DOF constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from src.assets.robots.unitree_g1.g1_constants import _update_assets as update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

WBOT_12DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "wbot" / "xmls" / "wbot_12dof.xml"
)
assert WBOT_12DOF_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, WBOT_12DOF_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(WBOT_12DOF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# Armature from ``doc/下半身转动惯量.xlsx`` (output-referred, kg·mm² → kg·m²).
##

ARMATURE_E2806 = 5607.889 / 1e6
ARMATURE_E4310 = 24222.8413745986 / 1e6
ARMATURE_E4315 = 32980.2813208958 / 1e6

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
# kp/kd scaled with new armature to keep prior √(kp/I) bandwidth.
E2806 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=18.6,
  damping=0.93,
  effort_limit=ACTUATOR_E2806.effort_limit,
  armature=ARMATURE_E2806,
)

E4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", ".*_ankle_pitch_joint"),
  stiffness=38.0,
  damping=1.52,
  effort_limit=ACTUATOR_E4310.effort_limit,
  armature=ARMATURE_E4310,
)
E4315 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=77.5,
  damping=2.07,
  effort_limit=ACTUATOR_E4315.effort_limit,
  armature=ARMATURE_E4315,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.5602089183583424),  # Home pose: lowest sole collision 1 mm above ground.
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": -0.02,
    "left_hip_roll_joint": 0.02,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    "right_ankle_roll_joint": 0.02,
    "left_ankle_roll_joint": -0.02,

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

WBOT_12DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E2806,
    E4310,
    E4315,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_wbot_12dof_robot_cfg() -> EntityCfg:
  """Get a fresh WBOT_12DOF robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=WBOT_12DOF_ARTICULATION,
  )


WBOT_12DOF_ACTION_SCALE: dict[str, float] = {}
for a in WBOT_12DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    WBOT_12DOF_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wbot_12dof_robot_cfg())

  viewer.launch(robot.spec.compile())
