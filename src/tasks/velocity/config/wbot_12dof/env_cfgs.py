"""WBOT 12DoF velocity environment configurations."""

from src.assets.robots import (
  WBOT_12DOF_ACTION_SCALE,
  get_wbot_12dof_robot_cfg,
)
from dataclasses import replace, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


WBOT_12DOF_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
assert len(WBOT_12DOF_JOINT_NAMES) == 12
WBOT_12DOF_JOINT_EXPR = r"(left|right)_.*_joint"
WBOT_12DOF_FOOT_SITES = ("left_foot", "right_foot")
WBOT_12DOF_FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def wbot_12dof_joint_asset_cfg() -> SceneEntityCfg:
  """All 12 actuated hinge joints, AMP/disc order."""
  return SceneEntityCfg(
    "robot",
    joint_names=WBOT_12DOF_JOINT_NAMES,
    preserve_order=True,
  )


def wbot_12dof_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create WBOT 12DoF rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  robot = get_wbot_12dof_robot_cfg()
  # Actuator lag is counted in physics steps: decimation=4 => 20 ms at 50 Hz.
  robot.articulation = replace(robot.articulation, actuators=tuple(
    replace(a, delay_min_lag=cfg.decimation, delay_max_lag=cfg.decimation)
    for a in robot.articulation.actuators
  ))
  cfg.scene.entities = {"robot": robot}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in WBOT_12DOF_FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
    fields=("found",),
    reduce="none",
    num_slots=1,
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
  joint_pos_action.scale = WBOT_12DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  from .commands import WbotVelocityCommandCfg
  twist_cmd = WbotVelocityCommandCfg(**{
    f.name: getattr(twist_cmd, f.name) for f in fields(twist_cmd)
  })
  twist_cmd.rel_standing_envs = 0.10
  twist_cmd.rel_forward_envs = 0.30
  cfg.commands["twist"] = twist_cmd
  twist_cmd.viz.z_offset = 0.6

  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["action_rate_l2"].weight = -0.5
  from src.tasks.velocity.mdp.rewards import energy_cost, stand_joint_vel_l2
  cfg.rewards["energy"] = RewardTermCfg(
    func=energy_cost, weight=-0.002,
    params={"asset_cfg": wbot_12dof_joint_asset_cfg()},
  )

  cfg.rewards["stand_joint_vel_l2"] = RewardTermCfg(
    func=stand_joint_vel_l2,
    weight=-0.05,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": wbot_12dof_joint_asset_cfg(),
    },
  )

  joint_cfg = wbot_12dof_joint_asset_cfg()
  cfg.observations["actor"].terms["joint_pos"].params = {"asset_cfg": joint_cfg}
  cfg.observations["actor"].terms["joint_vel"].params = {"asset_cfg": joint_cfg}
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": joint_cfg}
  cfg.observations["critic"].terms["joint_vel"].params = {"asset_cfg": joint_cfg}

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WBOT_12DOF_FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = WBOT_12DOF_FOOT_SITES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = WBOT_12DOF_FOOT_SITES
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def wbot_12dof_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create WBOT 12DoF flat terrain velocity configuration."""
  cfg = wbot_12dof_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.curriculum.pop("terrain_levels", None)

  # Positive, yaw-invariant torso upright reward; std=0.25 permits modest lean.
  from src.tasks.velocity.mdp.rewards import upright
  cfg.rewards["orientation"] = RewardTermCfg(
    func=upright,
    weight=1.0,
    params={
      "std": 0.25,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
