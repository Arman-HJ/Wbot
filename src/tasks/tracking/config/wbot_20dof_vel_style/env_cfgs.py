"""WBOT 20DoF velocity-conditioned whole-body style tracking.

Scaffold of "Driving the Style" (arXiv:2611.20888) on wbot:
- Dense BeyondMimic tracking rewards (no AMP discriminator).
- Actor sees planar velocity command + deployable proprioception.
- Critic keeps privileged body / reference terms.
- Standing-start + body-reward ramp curriculum.

Does not modify existing Kick / e1_mimic / AMP task configs.
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
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots import (
  WBOT_20DOF_SOFT_ANKLE_ACTION_SCALE,
  get_wbot_20dof_robot_cfg,
)
from src.assets.robots.wbot.wbot_20dof_constants import actuator_group_ids
from src.tasks.kick import mdp as kick_mdp
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

from .commands import VelocityConditionedMotionCommandCfg
from . import curriculums as vel_style_curriculums

_MOTIONS = (
  Path(__file__).resolve().parents[4] / "assets" / "motions" / "wbot_20dof" / "vel_style"
)
# Paper protocol: train generalization on random_walk only (see dataset.yaml).
_DEFAULT_MOTION_FILE = _MOTIONS / "random_walk_tracking.npz"

_PUSH_VELOCITY_RANGE = dict(VELOCITY_RANGE)
_RSI_POSE_RANGE = {
  "x": (-0.02, 0.02),
  "y": (-0.02, 0.02),
  "z": (-0.01, 0.01),
  "roll": (-0.12, 0.12),
  "pitch": (-0.12, 0.12),
  "yaw": (-0.15, 0.15),
}
_RSI_VELOCITY_RANGE = dict(VELOCITY_RANGE)
_RSI_JOINT_RANGE = (-0.12, 0.12)

# Paper-style sparse tracked bodies (pelvis as anchor).
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


def wbot_20dof_vel_style_env_cfg(
  play: bool = False,
  pd_gains: Literal[
    "armature_soft_ankle", "armature_soft_ankle_pace"
  ] = "armature_soft_ankle_pace",
  motion_file: Path | str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Velocity-conditioned tracking env (plain MLP deployable actor)."""
  pace = pd_gains == "armature_soft_ankle_pace"
  motion_path = Path(motion_file) if motion_file is not None else _DEFAULT_MOTION_FILE

  cfg = make_tracking_env_cfg()
  cfg.episode_length_s = 12.0
  cfg.sim.nconmax = 64
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=tracking_mdp.action_acc_l2,
    weight=-0.1,
  )

  # Standing-start body-tracking ramp (paper); no push harden curriculum yet.
  cfg.curriculum = {
    "standing_start_reward_ramp": CurriculumTermCfg(
      func=vel_style_curriculums.standing_start_reward_ramp,
      params={"command_name": "motion"},
    ),
  }

  cfg.scene.entities = {
    "robot": get_wbot_20dof_robot_cfg(
      action_delay_range=(0, 0) if play else (2, 8),
      pd_gains=pd_gains,
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

  old_motion = cfg.commands["motion"]
  cfg.commands["motion"] = VelocityConditionedMotionCommandCfg(
    entity_name=old_motion.entity_name,
    resampling_time_range=old_motion.resampling_time_range,
    debug_vis=old_motion.debug_vis,
    motion_file=str(motion_path),
    # Paper uses pelvis as the anchor body.
    anchor_body_name="pelvis",
    body_names=WBOT_TRACK_BODY_NAMES,
    sampling_mode="adaptive",
    start_time_step=0,
    start_time_step_prob=0.0,  # standing_start_prob covers start-from-rest
    pose_range=dict(_RSI_POSE_RANGE),
    velocity_range=dict(_RSI_VELOCITY_RANGE),
    joint_position_range=_RSI_JOINT_RANGE,
    viz=old_motion.viz,
    standing_start_prob=0.3,
    standing_stab_steps=30,
    standing_trans_steps=60,
    standing_body_weight_init=0.7,
  )

  # Deployable actor: velocity command (3) replaces ref joints.
  # Drop privileged lin-vel / anchor pos; keep ori error + projected gravity.
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

  # Critic: keep privileged robot body terms + EE world frames + motion EE.
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
  _ankle_joints = (".*_ankle_pitch_joint", ".*_ankle_roll_joint")
  _ankle_pitch_joints = (".*_ankle_pitch_joint",)

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
  cfg.viewer.body_name = "pelvis"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, VelocityConditionedMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.start_time_step_prob = 0.0
    motion_cmd.standing_start_prob = 0.0
    motion_cmd.debug_vis = True

  return cfg
