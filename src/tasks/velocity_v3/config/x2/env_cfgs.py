"""AgiBot X2 velocity v3 (high-speed v1) environment configurations."""

import math

from src.assets.robots import (
  AGIBOT_X2_CFG,
  AGIBOT_X2_NO_HEAD_JOINT_EXPR,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
# from mjlab.tasks.velocity import mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
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
from src.tasks.velocity_v3.velocity_env_cfg import make_velocity_v3_env_cfg
import src.tasks.velocity.mdp as mdp


def agibot_x2_velocity_v3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """AgiBot X2 full-body rough terrain velocity v3; lin_vel_x curriculum up to 3 m/s.

  Uses running nominal arm pose from ``x2_constants`` (shoulder pitch / elbow).
  """
  cfg = make_velocity_v3_env_cfg()
  cfg.scene.entities = {"robot": AGIBOT_X2_CFG}
  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

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
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  imu_torso_ang_vel_cfg = BuiltinSensorCfg(
    name="imu_torso_ang_vel",
    sensor_type="frameangvel",
    obj=ObjRef(type="body", name="torso_link", entity="robot"),
  )
  imu_lin_vel_cfg = BuiltinSensorCfg(
    name="imu_torso_lin_vel",
    sensor_type="framelinvel",
    obj=ObjRef(type="body", name="torso_link", entity="robot"),
  )
  torso_angmom_cfg = BuiltinSensorCfg(
    name="torso_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name="torso_link", entity="robot"),
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    imu_torso_ang_vel_cfg,
    imu_lin_vel_cfg,
    torso_angmom_cfg,
  )
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.viewer.body_name = "torso_link" 

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.50
  # Match 2026-05-28_16-42-25: heading-controlled yaw, no forward-only envs.
  twist_cmd.heading_command = True
  twist_cmd.rel_standing_envs = 0.1
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.ranges.heading = (-math.pi, math.pi)
  # Stage-0 command ranges before curriculum expands them.
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
        {"step": 20000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
        {"step": 40000 * 24, "lin_vel_x": (-1.5, 3.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
      ],
    },
  )

  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"].params["period"] = 0.6

  # 关节观测不包含头部关节（head_yaw_joint / head_pitch_joint）。
  no_head_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(AGIBOT_X2_NO_HEAD_JOINT_EXPR,),
  )
  cfg.observations["actor"].terms["joint_pos"].params = {"asset_cfg": no_head_joint_cfg}
  cfg.observations["actor"].terms["joint_vel"].params = {"asset_cfg": no_head_joint_cfg}
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": no_head_joint_cfg}
  cfg.observations["critic"].terms["joint_vel"].params = {"asset_cfg": no_head_joint_cfg}

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["torso_com_offset"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["torso_com_offset"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("torso_link",)

  # Events aligned with 2026-05-28_16-42-25 (no v3 encoder/IMU DR; add pelvis pushes).
  for key in ("reset_robot_joints", "external_force_torque", "encoder_bias", "imu_site_quat"):
    cfg.events.pop(key, None)
  cfg.events["push_robot"].interval_range_s = (3.0, 5.0)
  cfg.events["push_pelvis_root_velocity_lateral"] = EventTermCfg(
    func=mdp.push_by_setting_velocity_body_frame,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={
      "velocity_range": {
        "x": (0.0, 0.0),
        "y": (-1.0, 1.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
      "flag_standing_recovery": True,
    },
  )
  cfg.events["push_pelvis_root_velocity_longitudinal"] = EventTermCfg(
    func=mdp.push_by_setting_velocity_body_frame,
    mode="interval",
    interval_range_s=(7.0, 8.0),
    params={
      "velocity_range": {
        "x": (-1.0, 1.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
      "flag_standing_recovery": True,
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "operation": "abs",
      "ranges": (0.0, 0.2),
    },
  )

  cfg.terminations.pop("root_height_below_minimum", None)
  if "bad_orientation" in cfg.terminations:
    fell_over = cfg.terminations.pop("bad_orientation")
    cfg.terminations["fell_over"] = fell_over

  for reward_key, reward_func in (
    ("track_linear_velocity", mdp.track_linear_velocity),
    ("track_angular_velocity", mdp.track_angular_velocity),
    ("pose", mdp.variable_posture),
    ("torso_ang_vel_l2", mdp.body_angular_velocity_penalty),
    ("torso_angular_momentum", mdp.angular_momentum_penalty),
    ("feet_gait", mdp.feet_gait),
    ("feet_clearance", mdp.feet_clearance),
    ("feet_slip", mdp.feet_slip),
    ("soft_landing", mdp.soft_landing),
    ("stand_still", mdp.stand_still),
  ):
    if reward_key in cfg.rewards:
      cfg.rewards[reward_key].func = reward_func

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.05,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.05,
    # Waist.
    r".*waist_yaw.*": 0.1,
    r".*waist_roll.*": 0.05,
    r".*waist_pitch.*": 0.05,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
    # Head.
    r".*head_.*": 0.25,
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
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.3,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
    # Head.
    r".*head_.*": 0.25,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["torso_ang_vel_l2"].params["asset_cfg"].body_names = ("torso_link",)

  # Rewards aligned with logs/rsl_rl/x2_velocity_v3/2026-05-28_16-42-25.
  for key in ("base_height", "feet_swing_height", "electrical_power_cost"):
    cfg.rewards.pop(key, None)

  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["torso_angular_momentum"].weight = -0.05
  cfg.rewards["torso_ang_vel_l2"].weight = -0.5
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["feet_gait"].weight = 1.0
  cfg.rewards["feet_gait"].params["period"] = 0.6
  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["soft_landing"].weight = -1e-3
  cfg.rewards["stand_still"].weight = -1.0
  cfg.rewards["feet_clearance"].weight = -1.0
  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["command_name"] = "twist"
  cfg.rewards["feet_slip"].params["command_threshold"] = 0.1

  cfg.rewards["standing_waist_settle_exp"] = RewardTermCfg(
    func=mdp.standing_waist_settle_exp,
    weight=0.5,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "angvel_sensor_name": "robot/imu_torso_ang_vel",
      "gravity_sensor_name": "robot/imu_upvector",
      "w_omega": 1.0,
      "w_tilt": 1.5,
      "std_delta": 0.08,
      "k_fast": 0.05,
      "std_fast": 1.0,
    },
  )
  cfg.rewards["standing_push_recovery"] = RewardTermCfg(
    func=mdp.standing_push_two_step_recovery,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "min_lin_impulse": 0.5,
      "min_ang_impulse": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
      "target_steps": 2,
      "recovery_time_s": 2.5,
      "lin_vel_threshold": 0.12,
      "ang_vel_threshold": 0.45,
      "upright_threshold": 0.12,
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )


  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
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


def agibot_x2_velocity_v3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create AgiBot X2 flat terrain velocity v3 configuration."""
  cfg = agibot_x2_velocity_v3_rough_env_cfg(play=play)

  # 全足多段 capsule + 自碰等会抬高 nefc；300 会触发 mujoco_warp njmax overflow（需 >= nefc）
  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  # 基线可能已无 terrain_scan / height_scan；有则去掉，避免平面任务依赖射线高度。
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


def agibot_x2_velocity_v3_flat_enhanced_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat v3 secondary fine-tuning: backward disturbance + mass/delay DR for sim2real."""
  cfg = agibot_x2_velocity_v3_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.50

  # 根速度脉冲：向后线速度 + 后仰 pitch（mjlab push_by_setting_velocity）。
  cfg.events["push_robot"].func = mdp.push_by_setting_velocity_body_frame
  cfg.events["push_robot"].interval_range_s = (3.0, 5.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-1.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (-0.52, -0.10),
    "yaw": (0.0, 0.0),
  }

  return cfg


def agibot_x2_velocity_v3_full_body_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Alias: flat v3 already uses full-body X2 and full-body action scales."""
  return agibot_x2_velocity_v3_flat_env_cfg(play=play)
