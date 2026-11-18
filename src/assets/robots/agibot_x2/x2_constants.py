"""AgiBot X2 constants."""

import re
from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "agibot_x2" / "x2_assets" / "x2_ultra_simple.xml"
)
assert X2_XML.exists(), f"AgiBot X2 XML not found: {X2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  source_dir = (X2_XML.parent / "meshes").resolve()
  if not source_dir.is_dir():
    raise FileNotFoundError(
      f"AgiBot X2 meshes missing or broken symlink: {X2_XML.parent / 'meshes'} "
      f"(resolved={source_dir}). Sync real mesh files or fix the symlink."
    )
  for file in source_dir.rglob("*"):
    if not file.is_file():
      continue
    rel_path = file.relative_to(source_dir).as_posix()
    data = file.read_bytes()
    assets[rel_path] = data
    if meshdir:
      assets[f"{meshdir.rstrip('/')}/{rel_path}"] = data
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.66),
  joint_pos={
    # Slightly straighter stand (less crouch); hip+knee+ankle ≈ 0.
    ".*_hip_pitch_joint": -0.18,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.22,
    ".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.3600,
    ".*_shoulder_pitch_joint": 0.478,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": -1.67,
    ".*_wrist_yaw_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# SMP: arms straight (0 deg) at default; legs/waist/head match HOME_KEYFRAME.
HOME_KEYFRAME_SMP = EntityCfg.InitialStateCfg(
  pos=HOME_KEYFRAME.pos,
  joint_pos={
    **HOME_KEYFRAME.joint_pos,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
  },
  joint_vel=HOME_KEYFRAME.joint_vel,
)

AGIBOT_X2_NO_HEAD_JOINT_EXPR = r"^(?!head_(yaw|pitch)_joint$).*$"

##
# Collision config.
##

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

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

# armature / frictionloss from x2_ultra_simple.xml default joint.
X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"),
      stiffness=120.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_knee_joint",),
      stiffness=150.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint",),
      stiffness=40.0,
      damping=3.0,
      effort_limit=36.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_roll_joint",),
      stiffness=30.0,
      damping=2.0,
      effort_limit=24.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_yaw_joint",),
      stiffness=160.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
      stiffness=80.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_shoulder_pitch_joint",),
      stiffness=80.0,
      damping=4.0,
      effort_limit=36.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_shoulder_roll_joint",),
      stiffness=40.0,
      damping=1.0,
      effort_limit=36.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_shoulder_yaw_joint",),
      stiffness=40.0,
      damping=1.0,
      effort_limit=24.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_elbow_joint",),
      stiffness=40.0,
      damping=1.0,
      effort_limit=24.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_yaw_joint",),
      stiffness=40.0,
      damping=1.0,
      effort_limit=24.0,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
      stiffness=40.0,
      damping=1.0,
      effort_limit=2.2,
      armature=0.03,
      frictionloss=0.3,
      # delay_min_lag=2,
      # delay_max_lag=4,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

# scale = 0.25 * effort_limit / stiffness（与 G1/H2 一致）。
AGIBOT_X2_ACTION_SCALE: dict[str, float] = {}
for a in X2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    AGIBOT_X2_ACTION_SCALE[n] = 0.25 * e / s

AGIBOT_X2_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(),
  spec_fn=get_spec,
  articulation=X2_ARTICULATION,
)


def get_agibot_x2_robot_cfg() -> EntityCfg:
  """Return a fresh AgiBot X2 entity config (avoids shared mutable state)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(),
    spec_fn=get_spec,
    articulation=X2_ARTICULATION,
  )


def get_agibot_x2_smp_robot_cfg() -> EntityCfg:
  """AgiBot X2 with SMP default pose (straight arms)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME_SMP,
    collisions=(),
    spec_fn=get_spec,
    articulation=X2_ARTICULATION,
  )

# 12-DoF legs-only (upper body fixed in x2_fist.xml).
X2_FIST_XML: Path = (
  SRC_PATH / "assets" / "robots" / "agibot_x2" / "x2_assets" / "x2_fist.xml"
)
assert X2_FIST_XML.exists(), f"AgiBot X2 fist XML not found: {X2_FIST_XML}"

AGIBOT_X2_LEGS_ONLY_JOINT_EXPR = (
  r".*_hip_pitch_joint",
  r".*_hip_roll_joint",
  r".*_hip_yaw_joint",
  r".*_knee_joint",
  r".*_ankle_pitch_joint",
  r".*_ankle_roll_joint",
)

HOME_KEYFRAME_LEGS_ONLY = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.66),
  joint_pos={
    # Actuated legs (x2_fist / 12 DoF); slightly straighter stand.
    ".*_hip_pitch_joint": -0.18,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.22,
    ".*_ankle_roll_joint": 0.0,
    # Baked / unused on legs-only: exact names exported to deployment_alignment.
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.3600,
    "left_shoulder_pitch_joint": 0.478,
    "right_shoulder_pitch_joint": 0.478,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.67,
    "right_elbow_joint": -1.67,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

X2_LEGS_ONLY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"),
      stiffness=120.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      delay_min_lag=2,
      delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_knee_joint",),
      stiffness=150.0,
      damping=5.0,
      effort_limit=120.0,
      armature=0.03,
      frictionloss=0.3,
      delay_min_lag=2,
      delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint",),
      stiffness=40.0,
      damping=3.0,
      effort_limit=36.0,
      armature=0.03,
      frictionloss=0.3,
      delay_min_lag=2,
      delay_max_lag=4,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_roll_joint",),
      stiffness=30.0,
      damping=2.0,
      effort_limit=24.0,
      armature=0.03,
      frictionloss=0.3,
      delay_min_lag=2,
      delay_max_lag=4,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

AGIBOT_X2_LEGS_ONLY_ACTION_SCALE: dict[str, float] = {}
for a in X2_LEGS_ONLY_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    AGIBOT_X2_LEGS_ONLY_ACTION_SCALE[n] = 0.25 * e / s


def get_legs_only_spec() -> mujoco.MjSpec:
  """Load x2_fist (upper body fixed); name foot spheres + foot sole sites."""
  spec = mujoco.MjSpec.from_file(str(X2_FIST_XML))
  spec.assets = get_assets(spec.meshdir)
  foot_name_re = re.compile(r"^(left|right)_foot\d+_collision$")
  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    site_name = f"{side}_foot"
    existing = {s.name for s in spec.sites if s.name}
    if site_name not in existing:
      body.add_site(name=site_name, pos=[0.04, 0.0, -0.068])
    # Fist XML foot spheres are unnamed; name them for CollisionCfg / foot_friction DR.
    foot_idx = 0
    for geom in body.geoms:
      if geom.type != mujoco.mjtGeom.mjGEOM_SPHERE:
        continue
      foot_idx += 1
      geom.name = f"{side}_foot{foot_idx}_collision"
      geom.friction = [1.0, 0.005, 0.0001]
      geom.condim = 3
      geom.priority = 1
  # CollisionCfg cannot reliably disable unnamed geoms; turn off non-foot contacts here.
  for geom in spec.geoms:
    if foot_name_re.fullmatch(geom.name or ""):
      continue
    geom.contype = 0
    geom.conaffinity = 0
  return spec


# 12 spheres per foot in x2_fist.xml (named in get_legs_only_spec).
AGIBOT_X2_LEGS_ONLY_FOOT_GEOM_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 13)
)

AGIBOT_X2_LEGS_ONLY_CFG = EntityCfg(
  init_state=HOME_KEYFRAME_LEGS_ONLY,
  collisions=(FEET_ONLY_COLLISION,),
  spec_fn=get_legs_only_spec,
  articulation=X2_LEGS_ONLY_ARTICULATION,
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  print("AGIBOT_X2_ACTION_SCALE:")
  for pat, scale in AGIBOT_X2_ACTION_SCALE.items():
    print(f"  {pat!r}: {scale:.6g}")

  robot = Entity(AGIBOT_X2_CFG)
  viewer.launch(robot.spec.compile())
