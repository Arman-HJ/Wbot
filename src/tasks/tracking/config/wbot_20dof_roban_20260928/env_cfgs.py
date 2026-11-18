"""WBOT 20DoF pure motion-tracking (getup clip, no ball / no AMP)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots.wbot2_roban_20260928.wbot_20dof_constants import (
  WBOT_20DOF_ACTION_SCALE,
  get_wbot_20dof_robot_cfg,
)
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.history_obs import add_history_encoder_observations
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from . import terminations as local_terminations

_MOTION_FILE = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "motions"
  / "wbot_20dof"
  / "getup"
  / "prone_1_robans221_1011_wbot20.npz"
)

WBOT_TRACK_BODY_NAMES = (
  "pelvis",
  "torso_link",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
)

# 末端：双脚 + 双肘（与终止条件一致）
WBOT_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
)


def wbot_20dof_getup_tracking_env_cfg(
  *,
  play: bool = False,
  has_state_estimation: bool = False,
  pd_gains: Literal["armature", "armature_pace"] = "armature",
  history_encoder: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Pure tracking of Roban step-getup → wbot_20dof reference.

  Default is asymmetric / privileged-critic setup (``has_state_estimation=False``):
  actor drops ``motion_anchor_pos_b`` + ``base_lin_vel`` and gets
  ``projected_gravity``; critic keeps privileged body terms + world-frame EE.
  ``history_encoder`` adds ``actor_history`` / ``privileged`` groups (needs the
  ``HistoryActorCfg`` + ``ReconPpoAlgorithmCfg`` runner cfg).
  """
  cfg = make_tracking_env_cfg()
  cfg.terminations["anchor_pos"].func = local_terminations.bad_anchor_pos_z_only
  cfg.terminations["anchor_ori"].func = local_terminations.bad_anchor_ori
  cfg.terminations["ee_body_pos"].func = local_terminations.bad_motion_body_pos_z_only
  # Clip is ~17s at 50Hz; give a little slack before timeout.
  cfg.episode_length_s = 19.0
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512
  cfg.sim.contact_sensor_maxmatch = 500

  # Body–ground contact for prone push / getup.
  # PACE variants get per-limb command delay in training (legs 1-4, arms 3-7 steps).
  pace = pd_gains == "armature_pace"
  cfg.scene.entities = {
    "robot": get_wbot_20dof_robot_cfg(
      collision="getup",
      pd_gains=pd_gains,
      action_delay_range=(2, 8) if pace and not play else (0, 0),
    ),
  }

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
  motion_cmd.body_names = WBOT_TRACK_BODY_NAMES
  motion_cmd.sampling_mode = "adaptive"
  # 20% resets force clip start; RSI / pose noise unchanged.
  motion_cmd.start_time_step = 0
  motion_cmd.start_time_step_prob = 0.2
  motion_cmd.pose_range = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.01, 0.01),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.1, 0.1),
  }
  motion_cmd.velocity_range = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Root height/xy change a lot during getup — tighten root tracking.
  root_pos = cfg.rewards["motion_global_root_pos"]
  root_pos.weight = 1.5
  root_pos.params["std"] = 0.15
  root_ori = cfg.rewards["motion_global_root_ori"]
  root_ori.weight = 1.0
  root_ori.params["std"] = 0.35

  # 末端跟踪：脚踝 + 肘（相对锚点位置 / 姿态）
  cfg.rewards["motion_ee_pos"] = RewardTermCfg(
    func=tracking_mdp.motion_relative_body_position_error_exp,
    weight=1.5,
    params={
      "command_name": "motion",
      "std": 0.2,
      "body_names": WBOT_EE_BODY_NAMES,
    },
  )
  cfg.rewards["motion_ee_ori"] = RewardTermCfg(
    func=tracking_mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.35,
      "body_names": WBOT_EE_BODY_NAMES,
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = WBOT_EE_BODY_NAMES
  cfg.viewer.body_name = "torso_link"

  # --- Actor / critic observation layout ---
  # Default (has_state_estimation=False): deployable actor + privileged critic.
  actor_terms: dict[str, ObservationTermCfg] = {}
  drop_actor = (
    {"motion_anchor_pos_b", "base_lin_vel"}
    if not has_state_estimation
    else set()
  )
  for k, v in cfg.observations["actor"].terms.items():
    if k in drop_actor:
      continue
    actor_terms[k] = v
    if k == "base_ang_vel":
      actor_terms["projected_gravity"] = ObservationTermCfg(
        func=tracking_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      )
  if "projected_gravity" not in actor_terms:
    actor_terms["projected_gravity"] = ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  critic_terms: dict[str, ObservationTermCfg] = {}
  for k, v in cfg.observations["critic"].terms.items():
    critic_terms[k] = v
    if k == "base_ang_vel":
      critic_terms["projected_gravity"] = ObservationTermCfg(
        func=tracking_mdp.projected_gravity,
      )
  if "projected_gravity" not in critic_terms:
    critic_terms["projected_gravity"] = ObservationTermCfg(
      func=tracking_mdp.projected_gravity,
    )
  _ee_params = {
    "command_name": "motion",
    "body_names": WBOT_EE_BODY_NAMES,
  }
  critic_terms["ee_pos_w"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_pos_w, params=_ee_params
  )
  critic_terms["ee_ori_w"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_ori_w, params=_ee_params
  )
  critic_terms["motion_ee_pos_w"] = ObservationTermCfg(
    func=tracking_mdp.motion_body_pos_w, params=_ee_params
  )
  critic_terms["motion_ee_ori_w"] = ObservationTermCfg(
    func=tracking_mdp.motion_body_ori_w, params=_ee_params
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
  if history_encoder:
    # No PD-gain / mass DR in this task → those privileged targets would be constant.
    add_history_encoder_observations(
      cfg,
      foot_geoms=r"^(left|right)_foot[1-7]_collision$",
      play=play,
      exclude_privileged=("pd_gain_scale", "torso_mass_scale"),
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.start_time_step_prob = 0.0

  return cfg
