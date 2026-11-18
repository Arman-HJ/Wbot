"""AgiBot X2 legs-only (12-DoF) standing environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots import (
  AGIBOT_X2_LEGS_ONLY_ACTION_SCALE,
  AGIBOT_X2_LEGS_ONLY_CFG,
  AGIBOT_X2_LEGS_ONLY_FOOT_GEOM_NAMES,
  AGIBOT_X2_LEGS_ONLY_JOINT_EXPR,
)
from src.tasks.standing.standing_env_cfg import make_standing_env_cfg
import src.tasks.velocity.mdp as mdp


def _apply_x2_blind_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("height_scan", None)
    cfg.observations[group_name].terms.pop("measured_base_height", None)


def agibot_x2_standing_legs_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """AgiBot X2 12-DoF legs-only standing (upper body fixed)."""
  cfg = make_standing_env_cfg()
  cfg.scene.entities = {"robot": AGIBOT_X2_LEGS_ONLY_CFG}
  cfg.commands.pop("upper_body_motion", None)
  cfg.rewards.pop("tracking_upper_body_dofs", None)
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("ref_upper_dof_pos", None)
  site_names = ("left_foot", "right_foot")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame = ObjRef(type="body", name="torso_link", entity="robot")
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 56

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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = AGIBOT_X2_LEGS_ONLY_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # Keep base command ranges + command_vel curriculum for light velocity tracking.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.50
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  height_cmd = cfg.commands.get("base_height")
  if height_cmd is not None:
    height_cmd.ranges.height = (0.35, 0.72)
    height_cmd.default_height = 0.66
  if "track_base_height" in cfg.rewards:
    cfg.rewards["track_base_height"].params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=("torso_link",)
    )

  legs_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=AGIBOT_X2_LEGS_ONLY_JOINT_EXPR,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params = {"asset_cfg": legs_joint_cfg}
    cfg.observations[group].terms["joint_vel"].params = {"asset_cfg": legs_joint_cfg}

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    AGIBOT_X2_LEGS_ONLY_FOOT_GEOM_NAMES
  )
  if "torso_com_offset" in cfg.events:
    cfg.events["torso_com_offset"].params["asset_cfg"].body_names = ("torso_link",)
  if "base_mass" in cfg.events:
    cfg.events["base_mass"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["asset_cfg"] = legs_joint_cfg
  cfg.rewards["stand_still"].params["asset_cfg"] = legs_joint_cfg
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_running"] = {".*": 0.05}

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["torso_ang_vel_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  _apply_x2_blind_obs(cfg)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.sim.nconmax = 128
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_pelvis_root_velocity_lateral", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 7
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def agibot_x2_standing_legs_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """AgiBot X2 12-DoF legs-only standing on flat plane."""
  cfg = agibot_x2_standing_legs_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
