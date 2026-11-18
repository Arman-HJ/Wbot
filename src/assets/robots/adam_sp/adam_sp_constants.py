"""PND Adam SP constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ADAM_SP_XML: Path = SRC_PATH / "assets" / "robots" / "adam_sp" / "xml" / "adam_sp.xml"
ADAM_SP_MESH_DIR: Path = ADAM_SP_XML.parent.parent / "meshes"
assert ADAM_SP_XML.exists(), f"Adam SP XML not found: {ADAM_SP_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  _update_assets(assets, ADAM_SP_MESH_DIR, meshdir)
  return assets


def _update_assets(assets: dict[str, bytes], source_dir: Path, meshdir: str) -> None:
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
  spec = mujoco.MjSpec.from_file(str(ADAM_SP_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Command delay in physics timesteps (aligned with engineai PM01 ACTUATOR_DELAY_RANGE).
ADAM_SP_ACTUATOR_DELAY_RANGE = (2, 8)

# Motor specs: only PD (stiffness/damping) updated from provided table.

ADAM_SP_HIP_PITCH_VELOCITY_LIMIT = 15.0
ADAM_SP_HIP_ROLL_VELOCITY_LIMIT = 8.0
ADAM_SP_HIP_YAW_VELOCITY_LIMIT = 8.0
ADAM_SP_KNEE_VELOCITY_LIMIT = 15.0

ADAM_SP_HIP_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=300.0,
  damping=7.0,
  effort_limit=230.0,
  armature=0.13426,
  viscous_damping=0.0993125,
  frictionloss=1.589,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=300.0,
  damping=7.0,
  effort_limit=230.0,
  armature=0.13426,
  viscous_damping=0.0993125,
  frictionloss=0.8,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
# Back-compat alias (hip pitch + knee previously shared one cfg).
ADAM_SP_HIPPITCH_KNEEPITCH_ACTUATOR = ADAM_SP_HIP_PITCH_ACTUATOR
ADAM_SP_HIPROLL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
  ),
  stiffness=600.0,
  damping=10.0,
  effort_limit=160.0,
  # armature=0.281573,
  armature=0.762093,
  viscous_damping=1.054,
  frictionloss=4.216,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_HIPYAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
  ),
  stiffness=300.0,
  damping=2.0,
  effort_limit=105.0,
  # armature=0.23409,
  armature=0.081,
  viscous_damping=0.47175,
  frictionloss=3.774,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_ANKLEPITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
  ),
  stiffness=130.0,
  damping=3.5,
  effort_limit=40.0,
  armature=0.0549 * 2,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_ANKLEROLL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_roll_joint",
  ),
  stiffness=70.0,
  damping=2.0,
  effort_limit=12.0,
  armature=0.0549 * 2,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
  ),
  stiffness=400.0,
  damping=11.0,
  effort_limit=150.0,
  armature=0.23409,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_ARM_SHOULDER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=60.0,
  damping=3.0,
  effort_limit=65.0,
  armature=0.01,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_ARM_ELBOW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
  ),
  stiffness=60.0,
  damping=3.0,
  effort_limit=30.0,
  armature=0.01,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)
ADAM_SP_ARM_WRIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.4,
  armature=0.01,
  delay_min_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[0],
  delay_max_lag=ADAM_SP_ACTUATOR_DELAY_RANGE[1],
)



##
# Keyframe config (tall stance; matches Jul-2026 AdamSP-AMP runs).
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.95),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.00,
    "right_hip_roll_joint": 0.00,
    "left_hip_yaw_joint": -0.08,
    "right_hip_yaw_joint": 0.08,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": 0.00,
    "right_ankle_roll_joint": 0.00,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": -0.3,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
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
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Feet + leg capsules/shins: enables left-right leg self-collision while keeping
# torso/arms/pelvis collision geoms disabled (lighter than FULL_COLLISION).
_LEGS_FEET_COLLISION_EXPR = (
  r"^(left|right)_foot\d+_collision$",
  r"^(left|right)_(hip|thigh|shin).*_collision$",
)
LEGS_FEET_COLLISION = CollisionCfg(
  geom_names_expr=_LEGS_FEET_COLLISION_EXPR,
  contype={
    r"^(left|right)_foot\d+_collision$": 0,
    r"^(left|right)_(hip|thigh|shin).*_collision$": 1,
  },
  conaffinity=1,
  condim={
    r"^(left|right)_foot\d+_collision$": 3,
    r"^(left|right)_(hip|thigh|shin).*_collision$": 1,
  },
  priority={
    r"^(left|right)_foot\d+_collision$": 1,
    r"^(left|right)_(hip|thigh|shin).*_collision$": 0,
  },
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

##
# Final config.
##

ADAM_SP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_SP_HIP_PITCH_ACTUATOR,
    ADAM_SP_KNEE_ACTUATOR,
    ADAM_SP_HIPROLL_ACTUATOR,
    ADAM_SP_HIPYAW_ACTUATOR,
    ADAM_SP_ANKLEPITCH_ACTUATOR,
    ADAM_SP_ANKLEROLL_ACTUATOR,
    ADAM_SP_WAIST_ACTUATOR,
    ADAM_SP_ARM_SHOULDER_ACTUATOR,
    ADAM_SP_ARM_ELBOW_ACTUATOR,
    ADAM_SP_ARM_WRIST_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Legs-only (12 DoF): waist/arms baked rigid in the MjSpec.
ADAM_SP_LEGS_12DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_SP_HIP_PITCH_ACTUATOR,
    ADAM_SP_KNEE_ACTUATOR,
    ADAM_SP_HIPROLL_ACTUATOR,
    ADAM_SP_HIPYAW_ACTUATOR,
    ADAM_SP_ANKLEPITCH_ACTUATOR,
    ADAM_SP_ANKLEROLL_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

LEGS_12DOF_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.95),
  joint_pos={
    # Actuated legs (tall stance; matches Jul-2026 12DoF runs).
    ".*_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.00,
    "right_hip_roll_joint": 0.00,
    "left_hip_yaw_joint": -0.08,
    "right_hip_yaw_joint": 0.08,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": 0.00,
    "right_ankle_roll_joint": 0.00,
    # Baked / unused on 12DoF: exact names exported to deployment_alignment.
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.3,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.3,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Per-joint action scale aligned with engineai PM01 JointPositionActionCfg.scale.
ADAM_SP_ACTION_SCALE: dict[str, float] = {
  ".*_hip_pitch_joint": 0.5,
  ".*_hip_roll_joint": 0.2,
  ".*_hip_yaw_joint": 0.2,
  ".*_knee_joint": 0.5,
  ".*_ankle_pitch_joint": 0.5,
  ".*_ankle_roll_joint": 0.2,
  "waist_.*": 0.2,
  ".*_shoulder_.*": 0.2,
  ".*_elbow_.*": 0.2,
  ".*_wrist_.*": 0.2,
}

# pnd_rl_gym adam_lite_12dof uses uniform action_scale=0.5 on the 12 leg joints.
ADAM_SP_LEGS_12DOF_ACTION_SCALE: dict[str, float] = {
  ".*_hip_pitch_joint": 0.5,
  ".*_hip_roll_joint": 0.5,
  ".*_hip_yaw_joint": 0.5,
  ".*_knee_joint": 0.5,
  ".*_ankle_pitch_joint": 0.5,
  ".*_ankle_roll_joint": 0.5,
}


def get_adam_sp_robot_cfg() -> EntityCfg:
  """Get a fresh Adam SP robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ADAM_SP_ARTICULATION,
  )


def get_adam_sp_legs_12dof_spec() -> mujoco.MjSpec:
  """12-DoF legs-only: waist/arms baked rigid at HOME_KEYFRAME."""
  from src.assets.robots.adam_sp.adam_sp_legs_spec import build_adam_sp_legs_12dof_spec

  spec = build_adam_sp_legs_12dof_spec()
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_adam_sp_legs_12dof_robot_cfg() -> EntityCfg:
  """12-DoF Adam SP: only leg joints actuated; upper body frozen in pose."""
  return EntityCfg(
    init_state=LEGS_12DOF_KEYFRAME,
    collisions=(LEGS_FEET_COLLISION,),
    spec_fn=get_adam_sp_legs_12dof_spec,
    articulation=ADAM_SP_LEGS_12DOF_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  print("ADAM_SP_ACTUATOR_DELAY_RANGE:", ADAM_SP_ACTUATOR_DELAY_RANGE)
  print("ADAM_SP_ACTION_SCALE:")
  for pat, scale in ADAM_SP_ACTION_SCALE.items():
    print(f"  {pat!r}: {scale:.6g}")

  robot = Entity(get_adam_sp_robot_cfg())

  viewer.launch(robot.spec.compile())
