"""WBOT 20DoF flat tracking environment configurations."""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.assets.robots import WBOT_20DOF_ACTION_SCALE, get_wbot_20dof_robot_cfg
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_MOTION_FILE = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "motions"
  / "wbot_20dof"
  / "prone_1_robans221_1011_tuned.npz"
)

# Hands + feet end-effectors for get-up contact tracking.
_EE_BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
)


def wbot_20dof_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create WBOT 20DoF flat tracking for Roban prone get-up (tuned contacts).

  Uses weakened root tracking and boosted hand/foot end-effector rewards.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.num_envs = 4096
  cfg.scene.entities = {"robot": get_wbot_20dof_robot_cfg()}
  # Lying uses many body-ground contacts.
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 500
  # Clip is 393 frames at 50 Hz.
  cfg.episode_length_s = 7.86

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = WBOT_20DOF_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(_MOTION_FILE)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.sampling_mode = "adaptive"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(left|right)_foot[1-7]_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Weaken root/anchor tracking; emphasize hand/foot EE tracking for get-up contacts.
  cfg.rewards["motion_global_root_pos"].weight = 0.5
  cfg.rewards["motion_global_root_ori"].weight = 0.5
  cfg.rewards["motion_ee_pos"] = RewardTermCfg(
    func=tracking_mdp.motion_relative_body_position_error_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": 0.2,
      "body_names": _EE_BODIES,
    },
  )
  cfg.rewards["motion_ee_lin_vel"] = RewardTermCfg(
    func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.8,
      "body_names": _EE_BODIES,
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = _EE_BODIES

  cfg.viewer.body_name = "torso_link"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
