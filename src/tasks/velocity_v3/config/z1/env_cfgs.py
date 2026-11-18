"""Z1 velocity v3 (high-speed v1-style) environment configurations."""

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots.z1.z1_constants import (
  Z1_ACTION_SCALE,
  Z1_IMU_ANG_VEL_SENSOR,
  Z1_IMU_LIN_VEL_SENSOR,
  Z1_IMU_SITE,
  Z1_LEGS_ACTUATOR_EXPR,
  Z1_LEGS_JOINT_EXPR,
  Z1_TORSO_ANGMOM_SENSOR,
  Z1_TORSO_BODY,
  Z1_TORSO_UPVECTOR_SENSOR,
  get_z1_legs_robot_cfg,
  z1_robot_sensor_name,
)
from src.tasks.velocity_v3.velocity_env_cfg import make_velocity_v3_env_cfg
import src.tasks.velocity.mdp as mdp


def _disable_all_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove physics/actuator DR, pushes, and obs noise for nominal sim debugging."""
  for name in (
    "push_pelvis_root_velocity_lateral",
    "push_pelvis_root_velocity_longitudinal",
  ):
    cfg.events.pop(name, None)

  reset = cfg.events.get("reset_base")
  if reset is not None:
    reset.params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.01, 0.01),
      "yaw": (0.0, 0.0),
    }
    reset.params["velocity_range"] = {}

  cfg.observations["actor"].enable_corruption = False
  if "critic" in cfg.observations:
    cfg.observations["critic"].enable_corruption = False


def _apply_z1_imu_obs_and_reward_naming(cfg: ManagerBasedRlEnvCfg) -> None:
  """Pelvis IMU for ang vel; torso FK frame for projected gravity / upright."""
  imu_ang_vel_key = z1_robot_sensor_name(Z1_IMU_ANG_VEL_SENSOR)
  imu_lin_vel_key = z1_robot_sensor_name(Z1_IMU_LIN_VEL_SENSOR)
  torso_angmom_key = z1_robot_sensor_name(Z1_TORSO_ANGMOM_SENSOR)
  torso_upvector_key = z1_robot_sensor_name(Z1_TORSO_UPVECTOR_SENSOR)
  torso_body_cfg = SceneEntityCfg("robot", body_names=(Z1_TORSO_BODY,))

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

  # settle = cfg.rewards.get("standing_base_velocity_l2")
  # if settle is not None:
    # settle.params["angvel_sensor_name"] = imu_ang_vel_key
    # settle.params["linvel_sensor_name"] = imu_lin_vel_key

  if "upright" in cfg.rewards:
    upright = cfg.rewards["upright"]
    upright.params["gravity_sensor_name"] = torso_upvector_key
    upright.params["asset_cfg"] = torso_body_cfg


def _apply_z1_legs_control(cfg: ManagerBasedRlEnvCfg) -> None:
  """15-DoF policy (legs+waist); arms/hands/neck rigid at Z1_LOCOMOTION_KEYFRAME."""
  cfg.scene.entities = {
    "robot": get_z1_legs_robot_cfg(),
  }

  legs_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(Z1_LEGS_JOINT_EXPR,),
  )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = Z1_LEGS_ACTUATOR_EXPR
  joint_pos_action.scale = Z1_ACTION_SCALE

  cfg.observations["actor"].terms["joint_pos"].params = {"asset_cfg": legs_joint_cfg}
  cfg.observations["actor"].terms["joint_vel"].params = {"asset_cfg": legs_joint_cfg}
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": legs_joint_cfg}
  cfg.observations["critic"].terms["joint_vel"].params = {"asset_cfg": legs_joint_cfg}

  cfg.rewards["pose"].params["asset_cfg"] = legs_joint_cfg
  cfg.rewards["stand_still"].params["asset_cfg"] = legs_joint_cfg
  if "joint_vel_l2" in cfg.rewards:
    cfg.rewards["joint_vel_l2"].params["asset_cfg"] = legs_joint_cfg
  if "energy" in cfg.rewards:
    cfg.rewards["energy"].params["asset_cfg"] = legs_joint_cfg
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hipPitch.*": 0.5,
    r".*hipRoll.*": 0.05,
    r".*hipYaw.*": 0.05,
    r".*kneePitch.*": 0.5,
    r".*anklePitch.*": 0.15,
    r".*ankleRoll.*": 0.1,
    r".*waistYaw.*": 0.15,
    r".*waistRoll.*": 0.1,
    r".*waistPitch.*": 0.05,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hipPitch.*": 0.5,
    r".*hipRoll.*": 0.25,
    r".*hipYaw.*": 0.25,
    r".*kneePitch.*": 0.5,
    r".*anklePitch.*": 0.25,
    r".*ankleRoll.*": 0.1,
    r".*waistYaw.*": 0.3,
    r".*waistRoll.*": 0.3,
    r".*waistPitch.*": 0.1,
  }

  # settle = cfg.rewards.get("standing_base_velocity_l2")
  # if settle is not None:
  #   settle.params["angvel_sensor_name"] = z1_robot_sensor_name(Z1_IMU_ANG_VEL_SENSOR)
  #   settle.params["linvel_sensor_name"] = z1_robot_sensor_name(Z1_IMU_LIN_VEL_SENSOR)


def _apply_z1_blind_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Blind locomotion: no pelvis terrain grid in policy obs (Flat/Rough share actor dim)."""
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def z1_velocity_v3_legs_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Z1 legs+waist rough terrain v3; blind actor obs (same dim as Flat)."""
  cfg = make_velocity_v3_env_cfg()
  site_names = ("left_foot", "right_foot")
  geom_names = (
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
    "left_foot4_collision",
    "left_foot5_collision",
    "left_foot6_collision",
    "left_foot7_collision",
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
    "right_foot4_collision",
    "right_foot5_collision",
    "right_foot6_collision",
    "right_foot7_collision",
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
      pattern=r"^(toeLeft|toeRight)$",
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
      pattern="hipPitchLeft",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="hipPitchRight",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=4,
    history_length=4,
  )
  imu_ang_vel_cfg = BuiltinSensorCfg(
    name=Z1_IMU_ANG_VEL_SENSOR,
    sensor_type="frameangvel",
    obj=ObjRef(type="site", name=Z1_IMU_SITE, entity="robot"),
  )
  imu_lin_vel_cfg = BuiltinSensorCfg(
    name=Z1_IMU_LIN_VEL_SENSOR,
    sensor_type="framelinvel",
    obj=ObjRef(type="site", name=Z1_IMU_SITE, entity="robot"),
  )
  torso_angmom_cfg = BuiltinSensorCfg(
    name=Z1_TORSO_ANGMOM_SENSOR,
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name=Z1_TORSO_BODY, entity="robot"),
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    imu_ang_vel_cfg,
    imu_lin_vel_cfg,
    torso_angmom_cfg,
  )
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  cfg.viewer.body_name = Z1_TORSO_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.55
  twist_cmd.rel_standing_envs = 0.30

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

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["torso_com_offset"].params["asset_cfg"].body_names = (Z1_TORSO_BODY,)
  # cfg.events["base_mass"].params["asset_cfg"].body_names = (Z1_TORSO_BODY,)

  _apply_z1_imu_obs_and_reward_naming(cfg)
  _apply_z1_legs_control(cfg)

  legs_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(Z1_LEGS_JOINT_EXPR,),
  )


  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  _disable_all_domain_randomization(cfg)
  _apply_z1_blind_obs(cfg)

  if play:
    cfg.sim.nconmax = 512
    cfg.sim.contact_sensor_maxmatch = 512
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_pelvis_root_velocity_lateral", None)
    cfg.events.pop("push_pelvis_root_velocity_longitudinal", None)
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

  return cfg


def z1_velocity_v3_legs_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Z1 legs+waist flat terrain velocity v3."""
  cfg = z1_velocity_v3_legs_rough_env_cfg(play=play)

  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  _apply_z1_blind_obs(cfg)

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_pelvis_root_velocity_lateral", None)
    cfg.events.pop("push_pelvis_root_velocity_longitudinal", None)
    cfg.curriculum = {}

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg
