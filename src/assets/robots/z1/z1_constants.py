"""Z1 robot constants for mjlab locomotion."""

from pathlib import Path

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# XML and assets.
##

Z1_XML: Path = SRC_PATH / "assets" / "robots" / "z1" / "assets" / "z1.xml"
assert Z1_XML.exists(), f"Z1 XML not found: {Z1_XML}"
Z1_LEGS_ONLY_XML: Path = (
  SRC_PATH / "assets" / "robots" / "z1" / "assets" / "z1_legs_only.xml"
)
assert Z1_LEGS_ONLY_XML.exists(), f"Z1 legs-only XML not found: {Z1_LEGS_ONLY_XML}"

_Z1_XML_LOGGED = False

Z1_IMU_SITE = "imu"
Z1_PELVIS_BODY = "pelvis"
Z1_TORSO_BODY = "torso"
Z1_TORSO_IMU_SITE = "torso_imu"
Z1_IMU_UPVECTOR_SENSOR = "imu_upvector"
Z1_TORSO_UPVECTOR_SENSOR = "torso_upvector"
Z1_IMU_ANG_VEL_SENSOR = "imu_ang_vel"
Z1_IMU_LIN_VEL_SENSOR = "imu_lin_vel"
Z1_PELVIS_ANGMOM_SENSOR = "pelvis_angmom"
Z1_TORSO_ANGMOM_SENSOR = "torso_angmom"


def z1_robot_sensor_name(short_name: str) -> str:
  """Scene sensor key, e.g. ``robot/imu_ang_vel``."""
  return f"robot/{short_name}"


def _log_z1_xml_once() -> None:
  global _Z1_XML_LOGGED
  if not _Z1_XML_LOGGED:
    print(f"[INFO] Z1 XML: {Z1_XML.resolve()}", flush=True)
    _Z1_XML_LOGGED = True


##
# Manual PD actuator config.
##


def _manual_pd(
  target_names_expr: tuple[str, ...],
  effort_limit: float,
  stiffness: float,
  damping: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
  )


Z1_HIP_PITCH = _manual_pd(("hipPitch_.*",), 230.0, 305.0, 6.1)
Z1_KNEE = _manual_pd(("kneePitch_.*",), 230.0, 305.0, 6.1)
Z1_HIP_ROLL = _manual_pd(("hipRoll_.*",), 160.0, 700.0, 30.0)
Z1_HIP_YAW = _manual_pd(("hipYaw_.*",), 105.0, 405.0, 6.1)
Z1_ANKLE_PITCH = _manual_pd(("anklePitch_.*",), 40.0, 40.0, 0.8)
Z1_ANKLE_ROLL = _manual_pd(("ankleRoll_.*",), 40.0, 40.0, 0.8)
Z1_WAIST_YAW = _manual_pd(("waistYaw",), 110.0, 205.0, 15.0)
Z1_WAIST_ROLL = _manual_pd(("waistRoll",), 110.0, 405.0, 40.0)
Z1_WAIST_PITCH = _manual_pd(("waistPitch",), 110.0, 405.0, 30.0)

##
# Keyframes / control.
##

_STANDING_LEG_POS = {
  "hipPitch_Left": -0.20,
  "hipRoll_Left": 0.08,
  "hipYaw_Left": -0.08,
  "kneePitch_Left": 0.40,
  "anklePitch_Left": -0.20,
  "ankleRoll_Left": -0.08,
  "hipPitch_Right": -0.20,
  "hipRoll_Right": -0.08,
  "hipYaw_Right": 0.08,
  "kneePitch_Right": 0.40,
  "anklePitch_Right": -0.20,
  "ankleRoll_Right": 0.08,
  "waistRoll": 0.0,
  "waistPitch": 0.0,
  "waistYaw": 0.0,
}

_STRAIGHT_ARM_POS = {
  "shoulderPitch_Left": 0.0,
  "shoulderRoll_Left": 0.2,
  "shoulderYaw_Left": 0.0,
  "elbow_Left": -0.6,
  "wristYaw_Left": 0.0,
  "wristPitch_Left": 0.0,
  "wristRoll_Left": 0.0,
  "shoulderPitch_Right": 0.0,
  "shoulderRoll_Right": -0.2,
  "shoulderYaw_Right": 0.0,
  "elbow_Right": -0.6,
  "wristYaw_Right": 0.0,
  "wristPitch_Right": 0.0,
  "wristRoll_Right": 0.0,
}

Z1_LOCOMOTION_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={**_STANDING_LEG_POS, **_STRAIGHT_ARM_POS},
  joint_vel={".*": 0.0},
)

Z1_LEGS_JOINT_EXPR = (
  r"^(hipPitch_|hipRoll_|hipYaw_|kneePitch_|anklePitch_|ankleRoll_|waist).*"
)

Z1_LEGS_ACTUATOR_EXPR = (
  r"hipPitch_.*",
  r"hipRoll_.*",
  r"hipYaw_.*",
  r"kneePitch_.*",
  r"anklePitch_.*",
  r"ankleRoll_.*",
  r"waistYaw",
  r"waistRoll",
  r"waistPitch",
)

Z1_LEGS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    Z1_HIP_PITCH,
    Z1_KNEE,
    Z1_HIP_ROLL,
    Z1_HIP_YAW,
    Z1_ANKLE_PITCH,
    Z1_ANKLE_ROLL,
    Z1_WAIST_YAW,
    Z1_WAIST_ROLL,
    Z1_WAIST_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)

Z1_ACTION_SCALE = 0.25

##
# Collision config (simplified capsules; deploy / explicit tuning).
##

Z1_LOCOMOTION_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(left|right)_foot[1-7]_collision$",
    r"^pelvis_collision$",
    r"^(left|right)_(thigh|shin|hip)_collision$",
    r"^(left|right)_shin_brace_collision$",
    r"^torso_collision$",
    r"^(left|right)_(upper_arm|forearm|wrist|hand)_collision$",
    r"^head_collision$",
  ),
  contype=1,
  conaffinity=1,
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    r".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


def get_z1_legs_waist_spec():
  """Legs+waist spec: arms/hands/neck baked rigid at Z1_LOCOMOTION_KEYFRAME."""
  from src.assets.robots.z1.z1_legs_spec import build_z1_legs_waist_spec

  return build_z1_legs_waist_spec()


def get_z1_legs_robot_cfg() -> EntityCfg:
  """15-DoF legs+waist; arms/hands/neck baked at Z1_LOCOMOTION_KEYFRAME."""
  _log_z1_xml_once()
  return EntityCfg(
    init_state=Z1_LOCOMOTION_KEYFRAME,
    collisions=(),
    spec_fn=get_z1_legs_waist_spec,
    articulation=Z1_LEGS_ARTICULATION,
  )
