"""WBOT 20DoF e1_mimic tracking envs (victory / dance / backflip).

Configs mirror ``Wbot-20Dof-Kick`` (BeyondMimic rewards, adaptive RSI, push DR,
feet_slide, foot-Z) but without the soccer ball.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots import (
  WBOT_20DOF_SOFT_ANKLE_ACTION_SCALE,
  get_wbot_20dof_robot_cfg,
)
from src.assets.robots.wbot.wbot_20dof_constants import actuator_group_ids
from src.tasks.kick import mdp as kick_mdp
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.history_obs import add_history_encoder_observations
from src.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

_MOTIONS = Path(__file__).resolve().parents[4] / "assets" / "motions" / "wbot_20dof" / "e1_mimic"

# Same ranges as kick (soft / hard).
_PUSH_VELOCITY_RANGE = dict(VELOCITY_RANGE)
_PUSH_VELOCITY_RANGE_HARD = {
  k: (lo * 1.4, hi * 1.4) for k, (lo, hi) in VELOCITY_RANGE.items()
}
_RSI_POSE_RANGE = {
  "x": (-0.02, 0.02),
  "y": (-0.02, 0.02),
  "z": (-0.01, 0.01),
  "roll": (-0.12, 0.12),
  "pitch": (-0.12, 0.12),
  "yaw": (-0.15, 0.15),
}
_RSI_POSE_RANGE_HARD = {
  "x": (-0.025, 0.025),
  "y": (-0.025, 0.025),
  "z": (-0.012, 0.012),
  "roll": (-0.15, 0.15),
  "pitch": (-0.15, 0.15),
  "yaw": (-0.18, 0.18),
}
_RSI_VELOCITY_RANGE = dict(VELOCITY_RANGE)
_RSI_VELOCITY_RANGE_HARD = {
  k: (lo * 1.15, hi * 1.15) for k, (lo, hi) in VELOCITY_RANGE.items()
}
_RSI_JOINT_RANGE = (-0.12, 0.12)
_RSI_JOINT_RANGE_HARD = (-0.14, 0.14)

# Sparse tracked bodies — same layout as kick / G1 tracking.
WBOT_TRACK_BODY_NAMES = (
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
WBOT_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
)
WBOT_LEG_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
)


def _make_e1_mimic_tracking_env_cfg(
  *,
  motion_file: Path,
  episode_length_s: float,
  play: bool = False,
  pd_gains: Literal[
    "armature_soft_ankle", "armature_soft_ankle_pace"
  ] = "armature_soft_ankle",
  history_encoder: bool = False,
  collision: Literal["full", "feet", "getup"] = "full",
) -> ManagerBasedRlEnvCfg:
  """``history_encoder`` adds ``actor_history`` / ``privileged`` groups (needs the
  ``HistoryActorCfg`` + ``ReconPpoAlgorithmCfg`` runner cfg).

  ``collision="getup"`` disables self-collision and keeps body–ground contact
  (needed for prone fall / get-up clips).
  """
  pace = pd_gains == "armature_soft_ankle_pace"
  cfg = make_tracking_env_cfg()
  cfg.episode_length_s = episode_length_s
  # Getup body contacts need a larger solver buffer than kick/dance.
  cfg.sim.nconmax = 128 if collision == "getup" else 64
  cfg.sim.njmax = 512 if collision == "getup" else cfg.sim.njmax
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=tracking_mdp.action_acc_l2,
    weight=-0.1,
  )
  cfg.curriculum = {
    "push_and_term": CurriculumTermCfg(
      func=kick_mdp.push_termination_curriculum,
      params={
        "stages": [
          {
            "step": 0,
            "velocity_range": _PUSH_VELOCITY_RANGE,
            "anchor_pos_threshold": 0.25,
            "anchor_ori_threshold": 0.8,
            "ee_body_pos_threshold": 0.25,
            "rsi_pose_range": _RSI_POSE_RANGE,
            "rsi_velocity_range": _RSI_VELOCITY_RANGE,
            "rsi_joint_position_range": _RSI_JOINT_RANGE,
          },
          {
            "step": 0,
            "velocity_range": _PUSH_VELOCITY_RANGE_HARD,
            "anchor_pos_threshold": 0.35,
            "anchor_ori_threshold": 1.1,
            "ee_body_pos_threshold": 0.35,
            "rsi_pose_range": _RSI_POSE_RANGE_HARD,
            "rsi_velocity_range": _RSI_VELOCITY_RANGE_HARD,
            "rsi_joint_position_range": _RSI_JOINT_RANGE_HARD,
          },
        ],
      },
    ),
  }

  cfg.scene.entities = {
    "robot": get_wbot_20dof_robot_cfg(
      action_delay_range=(0, 0) if play else (2, 8),
      pd_gains=pd_gains,
      collision=collision,
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
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = WBOT_20DOF_SOFT_ANKLE_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion_file)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = WBOT_TRACK_BODY_NAMES
  motion_cmd.sampling_mode = "adaptive"
  # 20% resets force clip start; RSI / push DR unchanged (same as kick).
  motion_cmd.start_time_step = 0
  motion_cmd.start_time_step_prob = 0.2
  motion_cmd.pose_range = dict(_RSI_POSE_RANGE)
  motion_cmd.velocity_range = dict(_RSI_VELOCITY_RANGE)
  motion_cmd.joint_position_range = _RSI_JOINT_RANGE

  # No-state-estimation actor + projected gravity (same as kick).
  actor_terms: dict[str, ObservationTermCfg] = {}
  for k, v in cfg.observations["actor"].terms.items():
    if k in ("motion_anchor_pos_b", "base_lin_vel"):
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
  _ee_params = {"command_name": "motion", "body_names": WBOT_EE_BODY_NAMES}
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

  _foot_geoms = r"^(left|right)_foot[1-7]_collision$"
  # Same ankle DR as kick: bias toward active balance.
  _ankle_joints = (".*_ankle_pitch_joint", ".*_ankle_roll_joint")
  _ankle_pitch_joints = (".*_ankle_pitch_joint",)
  if history_encoder:
    add_history_encoder_observations(cfg, foot_geoms=_foot_geoms, play=play)
  cfg.events["push_robot"] = EventTermCfg(
    func=tracking_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": _PUSH_VELOCITY_RANGE},
  )
  cfg.events.pop("push_impulse", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["encoder_bias"].params["bias_range"] = (
    (-0.06, 0.06) if pace else (-0.015, 0.015)
  )
  cfg.events["encoder_bias_ankle"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_joints),
      "bias_range": (-0.03, 0.03),
    },
  )
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_foot_geoms),
      "operation": "abs",
      "ranges": (0.3, 1.6),
      "shared_random": True,
    },
  )
  cfg.events["body_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "alpha_range": (-0.05, 0.05),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # pseudo_inertia rewrites body_ipos from defaults; base_com must run after it.
  cfg.events["base_com"] = cfg.events.pop("base_com")
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )
  # pd_gains indexes Entity._actuators groups (not MuJoCo joint actuators).
  _ankle_actuator_ids = actuator_group_ids(
    cfg.scene.entities["robot"].articulation, _ankle_joints
  )
  cfg.events["pd_gains_ankle"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=_ankle_actuator_ids),
      "kp_range": (0.6, 1.0),
      "kd_range": (0.6, 1.0),
      "operation": "scale",
    },
  )
  # Ankle-pitch dry friction DR around KICK_JOINT_FRICTIONLOSS=0.3.
  cfg.events["ankle_joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_pitch_joints),
      "operation": "scale",
      "ranges": (0.75, 1.25),
    },
  )
  cfg.events["ankle_joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_joints),
      "operation": "abs",
      "ranges": (0.0, 0.15),
    },
  )
  cfg.events["ankle_joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_joints),
      "operation": "scale",
      "ranges": (0.85, 1.2),
    },
  )
  if pace:
    # Ankle dynamics come from PACE; no extra ankle-only DR on top.
    for name in (
      "encoder_bias_ankle",
      "pd_gains_ankle",
      "ankle_joint_friction",
      "ankle_joint_damping",
      "ankle_joint_armature",
    ):
      del cfg.events[name]
  cfg.events["foot_height"] = EventTermCfg(
    mode="startup",
    func=dr.geom_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_foot_geoms),
      "operation": "add",
      "axes": [2],
      "ranges": (-0.003, 0.0),
      "shared_random": True,
    },
  )

  _leg = set(WBOT_LEG_BODY_NAMES)

  def _std_map(leg: float, other: float) -> dict[str, float]:
    return {n: (leg if n in _leg else other) for n in WBOT_TRACK_BODY_NAMES}

  cfg.rewards["motion_body_pos"].params["std"] = _std_map(0.28, 0.3)
  cfg.rewards["motion_body_ori"].params["std"] = _std_map(0.38, 0.4)
  cfg.rewards["motion_body_lin_vel"].params["std"] = _std_map(0.95, 1.0)
  cfg.rewards["motion_body_ang_vel"].params["std"] = _std_map(3.0, 3.14)

  cfg.rewards["motion_foot_pos_z"] = RewardTermCfg(
    func=kick_mdp.motion_foot_pos_z_error_exp,
    weight=3.0,
    params={
      "command_name": "motion",
      "std": 0.1,
      "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
    },
  )
  cfg.rewards["feet_slide"] = RewardTermCfg(
    func=kick_mdp.feet_slide,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = WBOT_EE_BODY_NAMES
  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.start_time_step_prob = 0.0

  return cfg


def wbot_20dof_victory2_tracking_env_cfg(
  play: bool = False,
  pd_gains: Literal[
    "armature_soft_ankle", "armature_soft_ankle_pace"
  ] = "armature_soft_ankle",
) -> ManagerBasedRlEnvCfg:
  return _make_e1_mimic_tracking_env_cfg(
    motion_file=_MOTIONS / "dance" / "victory2_tracking.npz",
    episode_length_s=11.0,
    play=play,
    pd_gains=pd_gains,
    history_encoder=True,
  )


def wbot_20dof_victory1_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_e1_mimic_tracking_env_cfg(
    motion_file=_MOTIONS / "dance" / "victory1_tracking.npz",
    episode_length_s=14.5,
    play=play,
  )


def wbot_20dof_mj_dance_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_e1_mimic_tracking_env_cfg(
    motion_file=_MOTIONS / "dance" / "MJ_dance_tracking.npz",
    episode_length_s=50.0,
    play=play,
  )


def wbot_20dof_backflip_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_e1_mimic_tracking_env_cfg(
    motion_file=_MOTIONS / "backflip" / "robot_backflip_soma_tracking.npz",
    episode_length_s=4.0,
    play=play,
  )


_FALLGETUP_MOTION = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "motions"
  / "wbot_20dof"
  / "getup"
  / "fallAndGetUp1_subject1_played_hold1s.npz"
)


def wbot_20dof_fallgetup1_tracking_env_cfg(
  play: bool = False,
  pd_gains: Literal[
    "armature_soft_ankle", "armature_soft_ankle_pace"
  ] = "armature_soft_ankle",
) -> ManagerBasedRlEnvCfg:
  """SuperMimic fallAndGetUp1_subject1 played clip (hold1s), kick-aligned tracking.

  Clip is ~5.8 s at 50 Hz (prone hold → getup → stand hold). Uses getup collision
  so the torso/arms can push against the floor.
  """
  return _make_e1_mimic_tracking_env_cfg(
    motion_file=_FALLGETUP_MOTION,
    episode_length_s=7.0,
    play=play,
    pd_gains=pd_gains,
    history_encoder=True,
    collision="getup",
  )
