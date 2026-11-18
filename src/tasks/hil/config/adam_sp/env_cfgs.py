"""PND Adam SP HIL environment configurations."""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from src.assets.robots.adam_sp.adam_sp_constants import get_adam_sp_robot_cfg
from src.tasks.amp_v2.config.adam_sp.env_cfgs import (
  ADAM_SP_AMP_MOTION_BODY_NAMES,
  _default_expert_vis_npz,
)

from src.tasks.amp_v2.mdp.commands import AmpExpertMotionCommandCfg, AmpExpertVisCommandCfg
from src.tasks.amp_v2.config.adam_sp.rl_cfg import DEFAULT_AMP_DATA_DIR
from src.tasks.amp_v2 import mdp as amp_v2_mdp
from src.tasks.hil.hil_env_cfg import make_hil_env_cfg
from src.tasks.hil.mdp.twist_command import HilTwistCommandCfg
from src.tasks.velocity_v3.config.adam_sp.env_cfgs import (
  ADAM_SP_ACTION_SCALE,
  ADAM_SP_FOOT_SOLE_SITES,
  ADAM_SP_IMU_ANG_VEL_SENSOR,
  ADAM_SP_IMU_LIN_VEL_SENSOR,
  ADAM_SP_LOCOMOTION_JOINT_EXPR,
  ADAM_SP_TORSO_BODY,
  ADAM_SP_TORSO_UPVECTOR_SENSOR,
  _apply_adam_sp_blind_obs,
  adam_sp_robot_sensor_name,
)
import src.tasks.velocity.mdp as mdp


def adam_sp_hil_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Adam SP flat terrain HIL (tracking + AMP velocity task)."""
  cfg = make_hil_env_cfg()
  cfg.scene.entities = {"robot": get_adam_sp_robot_cfg()}

  site_names = ADAM_SP_FOOT_SOLE_SITES
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10)
  )
  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = None

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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ADAM_SP_ACTION_SCALE

  cfg.viewer.body_name = ADAM_SP_TORSO_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, HilTwistCommandCfg)
  twist_cmd.viz.z_offset = 1.45

  locomotion_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=ADAM_SP_LOCOMOTION_JOINT_EXPR,
  )
  cfg.observations["actor"].terms["joint_pos"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["actor"].terms["joint_vel"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["critic"].terms["joint_pos"].params = {"asset_cfg": locomotion_joint_cfg}
  cfg.observations["critic"].terms["joint_vel"].params = {"asset_cfg": locomotion_joint_cfg}

  amp_v2_mdp.add_rsl_rl_amp_observation_groups(
    cfg,
    locomotion_joint_cfg,
    ang_vel_sensor_name=adam_sp_robot_sensor_name(ADAM_SP_IMU_ANG_VEL_SENSOR),
    num_disc_obs_steps=10,
  )
  ang_vel_sensor = adam_sp_robot_sensor_name(ADAM_SP_IMU_ANG_VEL_SENSOR)
  lin_vel_sensor = adam_sp_robot_sensor_name(ADAM_SP_IMU_LIN_VEL_SENSOR)
  gravity_sensor = adam_sp_robot_sensor_name(ADAM_SP_TORSO_UPVECTOR_SENSOR)
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    if "base_ang_vel" in terms:
      terms["base_ang_vel"].params["sensor_name"] = ang_vel_sensor
    if "base_lin_vel" in terms:
      terms["base_lin_vel"].params["sensor_name"] = lin_vel_sensor
    if "projected_gravity" in terms:
      terms["projected_gravity"].func = mdp.projected_gravity_from_sensor
      terms["projected_gravity"].params["sensor_name"] = gravity_sensor
  _apply_adam_sp_blind_obs(cfg)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, AmpExpertMotionCommandCfg)
  motion_cmd.anchor_body_name = ADAM_SP_TORSO_BODY
  motion_cmd.body_names = ADAM_SP_AMP_MOTION_BODY_NAMES

  init_hil = cfg.events.get("init_hil_training")
  if init_hil is not None and not play:
    init_hil.params["amp_data_path"] = str(DEFAULT_AMP_DATA_DIR)
    init_hil.params["joint_asset_cfg"] = locomotion_joint_cfg
    init_hil.params["reset_min_speed"] = 0.15

  reset_hil = cfg.events.get("reset_hil_from_amp_motion")
  if reset_hil is not None and not play:
    reset_hil.params["joint_asset_cfg"] = locomotion_joint_cfg

  cfg.terminations["track_bad_joint_pos"].params["asset_cfg"] = locomotion_joint_cfg

  expert_ref = cfg.commands.get("expert_ref")
  if isinstance(expert_ref, AmpExpertVisCommandCfg) and not play:
    default_npz = _default_expert_vis_npz()
    if default_npz is not None:
      expert_ref.motion_file = str(default_npz)
    expert_ref.debug_vis = True

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (ADAM_SP_TORSO_BODY,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    twist_cmd.ranges.lin_vel_x = (0.0, 1.5)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    cfg.terminations.pop("amp_fell_over", None)

  return cfg
