"""PND Adam SP velocity v3 environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots.adam_sp.adam_sp_constants import (
  ADAM_SP_ACTION_SCALE,
  get_adam_sp_robot_cfg,
)
from src.tasks.velocity_v3.velocity_env_cfg import make_velocity_v3_env_cfg
import src.tasks.velocity.mdp as mdp

# Robot-specific names (kept here; do not add to adam_sp_constants.py).
ADAM_SP_TORSO_BODY = "torso_link"
ADAM_SP_IMU_SITE = "imu_in_pelvis"
ADAM_SP_IMU_ANG_VEL_SENSOR = "imu_ang_vel"
ADAM_SP_IMU_LIN_VEL_SENSOR = "imu_lin_vel"
ADAM_SP_TORSO_UPVECTOR_SENSOR = "torso_upvector"
ADAM_SP_TORSO_ANGMOM_SENSOR = "torso_angmom"

ADAM_SP_LOCOMOTION_JOINT_EXPR = (
  r".*_hip_pitch_joint",
  r".*_hip_roll_joint",
  r".*_hip_yaw_joint",
  r".*_knee_joint",
  r".*_ankle_pitch_joint",
  r".*_ankle_roll_joint",
  r"waist_roll_joint",
  r"waist_pitch_joint",
  r"waist_yaw_joint",
  r".*_shoulder_pitch_joint",
  r".*_shoulder_roll_joint",
  r".*_shoulder_yaw_joint",
  r".*_elbow_joint",
  r".*_wrist_yaw_joint",
  r".*_wrist_pitch_joint",
  r".*_wrist_roll_joint",
)
ADAM_SP_ANKLE_JOINT_EXPR = (r".*_ankle_pitch_joint", r".*_ankle_roll_joint")
ADAM_SP_ARMS_JOINT_EXPR = (
  # r".*_shoulder_pitch_joint",
  r".*_shoulder_roll_joint",
  r".*_shoulder_yaw_joint",
  r".*_elbow_joint",
  r".*_wrist_yaw_joint",
  r".*_wrist_pitch_joint",
  r".*_wrist_roll_joint",
)
ADAM_SP_ARM_GAIT_JOINT_EXPR = (r".*_shoulder_pitch_joint",)
# Sole sites for foot clearance / height raycast (z aligned with foot collision bottom).
ADAM_SP_FOOT_SOLE_SITES = ("left_foot_sole", "right_foot_sole")
# Knee bodies (left, right) — same order as ADAM_SP_FOOT_SOLE_SITES for swing lateral reward.
ADAM_SP_KNEE_BODIES = ("left_knee_link", "right_knee_link")
# Foot bodies for yaw-alignment penalty (ankle roll link = foot frame).
ADAM_SP_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ADAM_SP_ARM_SWING_JOINT_NAMES = {
  "left_hip_pitch": "left_hip_pitch_joint",
  "right_hip_pitch": "right_hip_pitch_joint",
  "left_shoulder_pitch": "left_shoulder_pitch_joint",
  "right_shoulder_pitch": "right_shoulder_pitch_joint",
}


def _adam_sp_joint_vel_index(joint_name: str) -> int:
  """Map actuated joint name to ``asset.data.joint_vel`` column index."""
  from mjlab.entity.entity import Entity

  entity = Entity(get_adam_sp_robot_cfg())
  joint_ids, resolved = entity.find_joints(joint_name, preserve_order=True)
  if len(joint_ids) != 1 or resolved[0] != joint_name:
    raise ValueError(
      f"Expected one joint named {joint_name!r}, got {list(resolved)}"
    )
  return joint_ids[0]


def _adam_sp_arm_swing_joint_indices() -> dict[str, int]:
  return {
    key: _adam_sp_joint_vel_index(name)
    for key, name in ADAM_SP_ARM_SWING_JOINT_NAMES.items()
  }


def adam_sp_robot_sensor_name(short_name: str) -> str:
  return f"robot/{short_name}"


def _apply_adam_sp_imu_obs_and_reward_naming(cfg: ManagerBasedRlEnvCfg) -> None:
  """Adam SP: pelvis IMU for ang vel; torso_imu site FK for projected gravity / upright."""
  imu_ang_vel_key = adam_sp_robot_sensor_name(ADAM_SP_IMU_ANG_VEL_SENSOR)
  imu_lin_vel_key = adam_sp_robot_sensor_name(ADAM_SP_IMU_LIN_VEL_SENSOR)
  torso_angmom_key = adam_sp_robot_sensor_name(ADAM_SP_TORSO_ANGMOM_SENSOR)
  torso_upvector_key = adam_sp_robot_sensor_name(ADAM_SP_TORSO_UPVECTOR_SENSOR)
  torso_body_cfg = SceneEntityCfg("robot", body_names=(ADAM_SP_TORSO_BODY,))

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    if "torso_ang_vel" in terms:
      term = terms.pop("torso_ang_vel")
      term.params["sensor_name"] = imu_ang_vel_key
      terms["imu_ang_vel"] = term
    if "torso_lin_vel" in terms:
      term = terms.pop("torso_lin_vel")
      term.params["sensor_name"] = imu_lin_vel_key
      terms["imu_lin_vel"] = term
    if "projected_gravity" in terms:
      terms["projected_gravity"].params["sensor_name"] = torso_upvector_key

  if "torso_ang_vel_l2" in cfg.rewards:
    cfg.rewards["torso_ang_vel_l2"].params["asset_cfg"] = torso_body_cfg
  if "torso_angular_momentum" in cfg.rewards:
    cfg.rewards["torso_angular_momentum"].params["sensor_name"] = torso_angmom_key

  if "upright" in cfg.rewards:
    upright = cfg.rewards["upright"]
    upright.params["gravity_sensor_name"] = torso_upvector_key
    upright.params["asset_cfg"] = torso_body_cfg

  if "standing_waist_settle_exp" in cfg.rewards:
    settle = cfg.rewards["standing_waist_settle_exp"]
    settle.params["angvel_sensor_name"] = imu_ang_vel_key
    settle.params["gravity_sensor_name"] = torso_upvector_key


def _apply_adam_sp_blind_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Blind locomotion: no pelvis terrain grid in policy obs (Flat/Rough share actor dim)."""
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def adam_sp_velocity_v3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Adam SP 29-DoF full-body rough terrain velocity v3."""
  cfg = make_velocity_v3_env_cfg()
  cfg.scene.entities = {"robot": get_adam_sp_robot_cfg()}
  site_names = ADAM_SP_FOOT_SOLE_SITES
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10)
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = 64

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="left_hip_pitch_link",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="right_hip_pitch_link",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=4,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ADAM_SP_ACTION_SCALE

  cfg.viewer.body_name = ADAM_SP_TORSO_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.45
  twist_cmd.rel_standing_envs = 0.10

  cfg.events.pop("push_pelvis_root_velocity_lateral", None)
  cfg.events.pop("push_pelvis_root_velocity_longitudinal", None)

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
      ],
    },
  )

  locomotion_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=ADAM_SP_LOCOMOTION_JOINT_EXPR,
  )
  cfg.observations["actor"].terms["joint_pos"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["actor"].terms["joint_vel"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["critic"].terms["joint_vel"].params = {"asset_cfg": locomotion_joint_cfg}

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["torso_com_offset"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)
  cfg.events["imu_site_quat"].params["asset_cfg"].site_names = (ADAM_SP_IMU_SITE,)

  cfg.rewards["pose"].params["asset_cfg"] = locomotion_joint_cfg
  cfg.rewards["stand_still"].params["asset_cfg"] = locomotion_joint_cfg
  cfg.rewards["electrical_power_cost"].params["asset_cfg"] = locomotion_joint_cfg

  cfg.rewards["pose"].params["std_standing"] = { r".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.20,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*waist_yaw.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["pose"].weight = 1.5
  _apply_adam_sp_imu_obs_and_reward_naming(cfg)

  cfg.rewards["torso_ang_vel_l2"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)
  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["base_height"].params["target_height"] = 1.13
  cfg.rewards["base_height"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["feet_lateral_deviation"] = RewardTermCfg(
    func=mdp.feet_lateral_deviation,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=ADAM_SP_FOOT_BODIES),
      "local_forward_axis": (1.0, 0.0, 0.0),
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=mdp.feet_distance,
    weight=1.5,
    params={
      "target_dist": 0.25,
      "std": 0.05,
      "asset_cfg": SceneEntityCfg("robot", body_names=ADAM_SP_FOOT_BODIES),
    },
  )


  _apply_adam_sp_blind_obs(cfg)

  if play:
    cfg.sim.nconmax = 256
    cfg.sim.contact_sensor_maxmatch = 512
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_pelvis_root_velocity_lateral", None)
    cfg.events.pop("push_pelvis_root_velocity_longitudinal", None)
    cfg.events.pop("external_force_torque", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    twist_cmd.rel_forward_envs = 0.0
    twist_cmd.rel_standing_envs = 0

  return cfg


def adam_sp_velocity_v3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Adam SP flat terrain velocity v3."""
  cfg = adam_sp_velocity_v3_rough_env_cfg(play=play)

  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  _apply_adam_sp_blind_obs(cfg)

  return cfg
