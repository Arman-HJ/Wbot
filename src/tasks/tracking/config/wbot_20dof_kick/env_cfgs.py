"""WBOT 20DoF kick tracking (motion imitation + ball) environment configs.

Registered under ``src.tasks.tracking.config.wbot_20dof_kick``.
Ball / kick MDP helpers live in ``src.tasks.kick.mdp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco

from mjlab.entity import EntityCfg
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
from src.tasks.kick.mdp.commands import KickMotionCommandCfg
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.history_obs import add_history_encoder_observations
from src.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

# Instantaneous velocity push: interval (1.0, 3.0).
# Stage-0 = stock; stage-2 = mild harden (1.4× push) — 2× was too hard to recover.
_STEPS_PER_ITER = 24
KICK_PUSH_VELOCITY_RANGE = dict(VELOCITY_RANGE)
KICK_PUSH_VELOCITY_RANGE_HARD = {
  k: (lo * 1.4, hi * 1.4) for k, (lo, hi) in VELOCITY_RANGE.items()
}

# RSI at reset (motion command).
KICK_RSI_POSE_RANGE = {
  "x": (-0.02, 0.02),
  "y": (-0.02, 0.02),
  "z": (-0.01, 0.01),
  "roll": (-0.12, 0.12),
  "pitch": (-0.12, 0.12),
  "yaw": (-0.15, 0.15),
}
KICK_RSI_POSE_RANGE_HARD = {
  "x": (-0.025, 0.025),
  "y": (-0.025, 0.025),
  "z": (-0.012, 0.012),
  "roll": (-0.15, 0.15),
  "pitch": (-0.15, 0.15),
  "yaw": (-0.18, 0.18),
}
KICK_RSI_VELOCITY_RANGE = dict(VELOCITY_RANGE)
KICK_RSI_VELOCITY_RANGE_HARD = {
  k: (lo * 1.15, hi * 1.15) for k, (lo, hi) in VELOCITY_RANGE.items()
}
KICK_RSI_JOINT_RANGE = (-0.12, 0.12)
KICK_RSI_JOINT_RANGE_HARD = (-0.14, 0.14)

_MOTION_FILE = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "motions"
  / "wbot_20dof"
  / "gvhmr_kick"
  / "kick4_13500_from1s_ampstand.npz"
)


# Visual-only ball (not in rewards). Tuned in Viser; small domain noise.
_BALL_POS = (0.095, 0.621, 0.110)
_BALL_POS_NOISE = (0.02, 0.02, 0.01)

# Sparse tracked bodies, same layout as Mjlab G1 tracking (no wrist → elbow).
WBOT_KICK_BODY_NAMES = (
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

# G1 uses ankles + wrists; WBOT has no wrist → elbows.
WBOT_KICK_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
)

# Leg bodies share a slightly tighter tracking std than arms/core.
WBOT_KICK_LEG_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
)


def get_soccer_ball_spec(
  radius: float = 0.11,
  mass: float = 0.43,
) -> mujoco.MjSpec:
  """Free soccer ball (approx. size/mass from football T1.xml)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=_BALL_POS)
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius,),
    mass=mass,
    rgba=(1.0, 1.0, 1.0, 1.0),
    friction=(0.5, 0.005, 0.0001),
  )
  return spec


def wbot_20dof_kick_env_cfg(
  play: bool = False,
  pd_gains: Literal[
    "armature_soft_ankle", "armature_soft_ankle_pace"
  ] = "armature_soft_ankle",
) -> ManagerBasedRlEnvCfg:
  """kick3 BeyondMimic tracking + foot-Z tracking + adaptive RSI.

  Ball stays visual-only. Extra ``motion_foot_pos_z`` (weight 3) targets ankle
  height completion; redundant EE/kick-leg 3D terms are not used.

  ``armature_soft_ankle_pace`` keeps the kick 6 Hz kp/kd and action scale, uses
  the PACE-identified arm and leg dynamics, and (as in the AMP PACE task) the
  per-limb PACE command delay (legs 1-4, arms 3-7 physics steps) and arm
  encoder bias (±0.2 rad) from the robot config, plus ±0.06 rad encoder bias.
  """
  pace = pd_gains == "armature_soft_ankle_pace"
  cfg = make_tracking_env_cfg()
  # Motion ~5.7s (kick + 0.5s stand); keep episode slightly longer than clip.
  cfg.episode_length_s = 6.0
  cfg.sim.nconmax = 64
  cfg.sim.contact_sensor_maxmatch = 500
  # action_rate=-0.1; action_smoothness = AMP's action_acc_l2 but weight=-0.1.
  # Soft → hard curriculum. hard.step=0：接训自 ≥5000 ckpt 时本 run 直接二阶段
  # （新 env 的 common_step_counter 从 0 计，无法继承上一 run 的 step）。
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
            "velocity_range": KICK_PUSH_VELOCITY_RANGE,
            "anchor_pos_threshold": 0.25,
            "anchor_ori_threshold": 0.8,
            "ee_body_pos_threshold": 0.25,
            "rsi_pose_range": KICK_RSI_POSE_RANGE,
            "rsi_velocity_range": KICK_RSI_VELOCITY_RANGE,
            "rsi_joint_position_range": KICK_RSI_JOINT_RANGE,
          },
          {
            "step": 0,  # immediate stage-2 for resume-past-5000; last matching stage wins
            "velocity_range": KICK_PUSH_VELOCITY_RANGE_HARD,
            # Milder term loosen than before (was 0.50 / 1.6 / 0.50).
            "anchor_pos_threshold": 0.35,
            "anchor_ori_threshold": 1.1,
            "ee_body_pos_threshold": 0.35,
            "rsi_pose_range": KICK_RSI_POSE_RANGE_HARD,
            "rsi_velocity_range": KICK_RSI_VELOCITY_RANGE_HARD,
            "rsi_joint_position_range": KICK_RSI_JOINT_RANGE_HARD,
          },
        ],
      },
    ),
  }

  # Match AMP walk/run: armature PD + action delay (physics steps @ 5 ms → ~10–40 ms).
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
  cfg.commands["motion"] = KickMotionCommandCfg(
    entity_name=old_motion.entity_name,
    resampling_time_range=old_motion.resampling_time_range,
    debug_vis=old_motion.debug_vis,
    motion_file=str(_MOTION_FILE),
    anchor_body_name="torso_link",
    body_names=WBOT_KICK_BODY_NAMES,
    sampling_mode="adaptive",
    # 20% resets force clip start; RSI / push DR unchanged.
    start_time_step=0,
    start_time_step_prob=0.2,
    # RSI ranges; curriculum hardens after iter 5000.
    pose_range=dict(KICK_RSI_POSE_RANGE),
    velocity_range=dict(KICK_RSI_VELOCITY_RANGE),
    joint_position_range=KICK_RSI_JOINT_RANGE,
    viz=old_motion.viz,
    ball_placement="none",
  )

  # No-state-estimation actor (drop anchor pos + lin vel) + projected gravity.
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
  # 4 EE frames in world (robot + motion target). Existing body_pos/ori are
  # anchor-relative for all tracked links — not world-frame EE tracking.
  _ee_params = {
    "command_name": "motion",
    "body_names": WBOT_KICK_EE_BODY_NAMES,
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

  # Instantaneous velocity push; magnitude ramped by curriculum (interval fixed).
  _foot_geoms = r"^(left|right)_foot[1-7]_collision$"
  # Loose foot linkage: ankle-only DR. Bias toward active balance (soft PD /
  # light viscous), not passive structural lock (high kd / high dof damping).
  _ankle_joints = (".*_ankle_pitch_joint", ".*_ankle_roll_joint")
  _ankle_pitch_joints = (".*_ankle_pitch_joint",)
  add_history_encoder_observations(cfg, foot_geoms=_foot_geoms, play=play)

  cfg.events["push_robot"] = EventTermCfg(
    func=tracking_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": KICK_PUSH_VELOCITY_RANGE},
  )
  cfg.events.pop("push_impulse", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["encoder_bias"].params["bias_range"] = (
    (-0.06, 0.06) if pace else (-0.015, 0.015)
  )
  # 1) Larger ankle encoder gap → policy must actively correct mismatch.
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
  # 3) Soft ankle PD (mostly ≤1): cannot lock stance via stiff structure.
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
  # 2) Dry friction around KICK_JOINT_FRICTIONLOSS=0.3 (ankle pitch DR scale).
  cfg.events["ankle_joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_pitch_joints),
      "operation": "scale",
      "ranges": (0.75, 1.25),
    },
  )
  # 3b) Light viscous only — high dof_damping would passively lock the ankle.
  cfg.events["ankle_joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_joints),
      "operation": "abs",
      "ranges": (0.0, 0.15),
    },
  )
  # 4) Ankle armature scatter.
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
  # Same BeyondMimic body rewards; per-body std only (legs slightly tighter).
  _leg = set(WBOT_KICK_LEG_BODY_NAMES)

  def _std_map(leg: float, other: float) -> dict[str, float]:
    return {n: (leg if n in _leg else other) for n in WBOT_KICK_BODY_NAMES}

  cfg.rewards["motion_body_pos"].params["std"] = _std_map(0.28, 0.3)
  cfg.rewards["motion_body_ori"].params["std"] = _std_map(0.38, 0.4)
  cfg.rewards["motion_body_lin_vel"].params["std"] = _std_map(0.95, 1.0)
  cfg.rewards["motion_body_ang_vel"].params["std"] = _std_map(3.0, 3.14)

  # Foot height only (redundant 3D EE / kick-leg terms removed).
  cfg.rewards["motion_foot_pos_z"] = RewardTermCfg(
    func=kick_mdp.motion_foot_pos_z_error_exp,
    weight=3.0,
    params={
      "command_name": "motion",
      "std": 0.1,
      "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
    },
  )
  # Penalize ankle xy velocity while in ground contact (stance foot slip).
  cfg.rewards["feet_slide"] = RewardTermCfg(
    func=kick_mdp.feet_slide,
    weight=-0.25,  # same as AMP walk/run
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = WBOT_KICK_EE_BODY_NAMES

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, KickMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.start_time_step_prob = 0.0
    # Pin motion input at Hold Frame; Viser "Motion Clip → Play Clip" advances once.
    motion_cmd.loop = False
    motion_cmd.debug_vis = True

  return cfg
