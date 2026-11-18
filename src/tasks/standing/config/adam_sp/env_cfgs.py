"""PND Adam SP standing environment configurations."""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.robots.adam_sp.adam_sp_constants import (
  ADAM_SP_ACTION_SCALE,
  get_adam_sp_robot_cfg,
)
from src.tasks.standing.standing_env_cfg import make_standing_env_cfg
from src.tasks.standing.mdp.actions import ResidualUpperBodyJointPositionActionCfg
from src.tasks.standing.mdp.upper_body_motion import DEFAULT_UPPER_ARM_JOINT_NAMES
from src.tasks.velocity_v3.config.adam_sp.env_cfgs import (
  ADAM_SP_FOOT_SOLE_SITES,
  ADAM_SP_IMU_SITE,
  ADAM_SP_LOCOMOTION_JOINT_EXPR,
  ADAM_SP_TORSO_BODY,
  _apply_adam_sp_imu_obs_and_reward_naming,
)
import src.tasks.velocity.mdp as mdp


_ADAM_SP_UPPER_MOTION_DIR = (
  Path(__file__).resolve().parents[5] / "src/assets/upper_motions/pnd_adam_sp"
)


_ADAM_SP_LEG_IK = {
  "thigh_length": 0.4618,
  "shank_length": 0.3700,
  "torso_hip_offset": 0.2103,
}


def adam_sp_standing_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Adam SP 29-DoF standing on complex rough terrain with multi-type pushes."""
  cfg = make_standing_env_cfg()
  cfg.scene.entities = {"robot": get_adam_sp_robot_cfg()}
  site_names = ADAM_SP_FOOT_SOLE_SITES
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10)
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame = ObjRef(type="body", name=ADAM_SP_TORSO_BODY, entity="robot")
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 108

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
  cfg.actions["joint_pos"] = ResidualUpperBodyJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=ADAM_SP_ACTION_SCALE,
    use_default_offset=True,
    upper_body_joint_names=DEFAULT_UPPER_ARM_JOINT_NAMES,
    upper_body_command_name="upper_body_motion",
  )

  upper_motion = cfg.commands.get("upper_body_motion")
  if upper_motion is not None:
    upper_motion.motion_dir = str(_ADAM_SP_UPPER_MOTION_DIR)
    upper_motion.fix_upper_body_prob = 0.3 if not play else 0.0

  if "tracking_upper_body_dofs" in cfg.rewards:
    cfg.rewards["tracking_upper_body_dofs"].params["asset_cfg"] = SceneEntityCfg(
      "robot",
      joint_names=DEFAULT_UPPER_ARM_JOINT_NAMES,
    )

  cfg.viewer.body_name = ADAM_SP_TORSO_BODY

  # Keep base command ranges + command_vel curriculum for light velocity tracking.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.45
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None

  locomotion_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=ADAM_SP_LOCOMOTION_JOINT_EXPR,
  )
  waist_leg_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
      r".*_ankle_roll_joint",
      r"waist_roll_joint",
      r"waist_pitch_joint",
      r"waist_yaw_joint",
    ),
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params = {
      "asset_cfg": locomotion_joint_cfg
    }
    cfg.observations[group].terms["joint_vel"].params = {
      "asset_cfg": locomotion_joint_cfg
    }
    if "measured_base_height" in cfg.observations[group].terms:
      cfg.observations[group].terms["measured_base_height"].params["asset_cfg"] = (
        SceneEntityCfg("robot", body_names=(ADAM_SP_TORSO_BODY,))
      )

  height_cmd = cfg.commands.get("base_height")
  if height_cmd is not None:
    height_cmd.ranges.height = (0.55, 1.20)
    height_cmd.default_height = 1.13
  if "track_base_height" in cfg.rewards:
    cfg.rewards["track_base_height"].params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=(ADAM_SP_TORSO_BODY,)
    )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  if "torso_com_offset" in cfg.events:
    cfg.events["torso_com_offset"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)
  if "imu_site_quat" in cfg.events:
    cfg.events["imu_site_quat"].params["asset_cfg"].site_names = (ADAM_SP_IMU_SITE,)
  if "base_mass" in cfg.events:
    cfg.events["base_mass"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)

  cfg.rewards["pose"].params["asset_cfg"] = locomotion_joint_cfg
  cfg.rewards["stand_still"].params["asset_cfg"] = waist_leg_joint_cfg
  for key in ("pose", "stand_still"):
    cfg.rewards[key].params.update(_ADAM_SP_LEG_IK)
  if "reset_standing_height" in cfg.events:
    cfg.events["reset_standing_height"].params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=(ADAM_SP_TORSO_BODY,)
    )
    cfg.events["reset_standing_height"].params.update(_ADAM_SP_LEG_IK)

  # Rename torso/IMU sensor keys to Adam SP XML names; standing uses root
  # projected_gravity (no sensor), so strip the injected sensor_name.
  _apply_adam_sp_imu_obs_and_reward_naming(cfg)
  for group_name in ("actor", "critic"):
    pg = cfg.observations[group_name].terms.get("projected_gravity")
    if pg is not None:
      pg.params.pop("sensor_name", None)
  cfg.rewards["upright"].params.pop("gravity_sensor_name", None)

  cfg.rewards["torso_ang_vel_l2"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

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
    twist_cmd = cfg.commands["twist"]
    twist_cmd.rel_standing_envs = 1.0
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
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


def adam_sp_standing_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Adam SP 29-DoF standing on flat plane (upper-body FALCON tracking)."""
  cfg = adam_sp_standing_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  cfg.events.pop("randomize_terrain", None)

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("height_scan", None)
    measured = cfg.observations[group_name].terms.get("measured_base_height")
    if measured is not None:
      measured.params["terrain_sensor_name"] = None

  if "track_base_height" in cfg.rewards:
    cfg.rewards["track_base_height"].params["terrain_sensor_name"] = None

  return cfg
