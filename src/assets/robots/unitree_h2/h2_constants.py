"""Unitree H2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_h2" / "xmls" / "h2.xml"
)
assert H2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  _update_assets(assets, H2_XML.parent / "assets", meshdir)
  return assets


def _update_assets(assets: dict[str, bytes], source_dir: Path, meshdir: str) -> None:
  """Backwards-compatible asset collector for older mjlab versions."""
  if not source_dir.exists():
    return
  for file in source_dir.rglob("*"):
    if not file.is_file():
      continue
    rel_path = file.relative_to(source_dir).as_posix()
    assets[rel_path] = file.read_bytes()
    if meshdir:
      assets[f"{meshdir.rstrip('/')}/{rel_path}"] = assets[rel_path]


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

H2_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch.*",
    ".*_hip_roll.*",
    ".*_hip_yaw.*",
    ".*_knee.*",
  ),
  stiffness=200.0,
  damping=4.0,
  effort_limit=360.0,
  armature=0.01,
)
H2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=19.0,
  armature=0.01,
)
H2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=66.0,
  armature=0.01,
)
H2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_.*",
  ),
  stiffness=150.0,
  damping=3.0,
  effort_limit=120.0,
  armature=0.01,
)
H2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch.*",
    ".*_shoulder_roll.*",
    ".*_shoulder_yaw.*",
    ".*_elbow.*",
    ".*_wrist_roll.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=54.0,
  armature=0.01,
)
H2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch.*",
    ".*_wrist_yaw.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=25.0,
  armature=0.01,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.25,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
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

H2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2_ACTUATOR_LEG,
    H2_ACTUATOR_ANKLE_ROLL,
    H2_ACTUATOR_ANKLE_PITCH,
    H2_ACTUATOR_WAIST,
    H2_ACTUATOR_ARM,
    H2_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h2_robot_cfg() -> EntityCfg:
  """Get a fresh H2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H2_ARTICULATION,
  )


H2_ACTION_SCALE: dict[str, float] = {}
for a in H2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H2_ACTION_SCALE[n] = 0.25 * e / s


##
# Temporary motor set (peak torque only; kp / kd / armature unchanged).
#   hip_pitch / knee / hip_roll:          330 Nm
#   hip_yaw:                              120 Nm
#   ankle_pitch / ankle_roll:              70 Nm
#   shoulder_* / elbow / waist_yaw:        70 Nm
#   wrist_roll / wrist_pitch / wrist_yaw:   5 Nm
# waist_roll / waist_pitch are not in the list and keep 120 Nm.
##


def _pd(
  names: tuple[str, ...], stiffness: float, damping: float, effort: float
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=names,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort,
    armature=0.01,
  )


H2_TMP_MOTOR_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _pd((".*_hip_pitch.*", ".*_knee.*", ".*_hip_roll.*"), 200.0, 4.0, 330.0),
    _pd((".*_hip_yaw.*",), 200.0, 4.0, 120.0),
    _pd((".*_ankle_pitch.*",), 40.0, 2.0, 70.0),
    _pd((".*_ankle_roll.*",), 40.0, 2.0, 70.0),
    _pd(("waist_yaw.*",), 150.0, 3.0, 70.0),
    _pd(("waist_roll.*", "waist_pitch.*"), 150.0, 3.0, 120.0),
    _pd(
      (
        ".*_shoulder_pitch.*",
        ".*_shoulder_roll.*",
        ".*_shoulder_yaw.*",
        ".*_elbow.*",
      ),
      40.0,
      2.0,
      70.0,
    ),
    _pd((".*_wrist_roll.*",), 40.0, 2.0, 5.0),
    _pd((".*_wrist_pitch.*", ".*_wrist_yaw.*"), 20.0, 1.0, 5.0),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h2_tmp_motor_robot_cfg() -> EntityCfg:
  """H2 with the temporary motor torque limits (use with ``H2_ACTION_SCALE``)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H2_TMP_MOTOR_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h2_robot_cfg())

  viewer.launch(robot.spec.compile())
