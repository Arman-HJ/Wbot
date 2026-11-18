"""WBOT 12DoF kick (motion imitation + ball) environment configs."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import WBOT_12DOF_ACTION_SCALE, get_wbot_12dof_robot_cfg
from src.tasks.kick import mdp as kick_mdp
from src.tasks.kick.mdp.commands import KickMotionCommandCfg
from src.tasks.tracking import mdp as tracking_mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_MOTION_FILE = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "motions"
  / "wbot_12dof"
  / "gvhmr_kick"
  / "kick2_wbot_12dof.npz"
)

WBOT_KICK_BODY_NAMES = (
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
)


def get_soccer_ball_spec(
  radius: float = 0.11,
  mass: float = 0.43,
) -> mujoco.MjSpec:
  """Free soccer ball (approx. size/mass from football T1.xml)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(0.45, 0.0, radius))
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


def wbot_12dof_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Motion-imitation kick task (no AMP): track right-kick clip + kick ball forward."""
  cfg = make_tracking_env_cfg()
  cfg.episode_length_s = 8.0
  cfg.sim.nconmax = 64
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {
    "robot": get_wbot_12dof_robot_cfg(),
    "ball": EntityCfg(spec_fn=get_soccer_ball_spec),
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
  joint_pos_action.scale = WBOT_12DOF_ACTION_SCALE

  # Replace motion command with kick variant (places ball AFTER robot RSI).
  old_motion = cfg.commands["motion"]
  cfg.commands["motion"] = KickMotionCommandCfg(
    entity_name=old_motion.entity_name,
    resampling_time_range=old_motion.resampling_time_range,
    debug_vis=old_motion.debug_vis,
    motion_file=str(_MOTION_FILE),
    anchor_body_name="torso_link",
    body_names=WBOT_KICK_BODY_NAMES,
    sampling_mode="start",
    pose_range={
      "x": (-0.02, 0.02),
      "y": (-0.02, 0.02),
      "z": (-0.01, 0.01),
      "roll": (-0.05, 0.05),
      "pitch": (-0.05, 0.05),
      "yaw": (-0.1, 0.1),
    },
    velocity_range=old_motion.velocity_range,
    joint_position_range=old_motion.joint_position_range,
    viz=old_motion.viz,
  )

  # Ball state is privileged (critic only); actor does not see ball pose/vel.
  cfg.observations["critic"].terms["ball_pos_b"] = ObservationTermCfg(
    func=kick_mdp.ball_pos_b,
    params={"ball_name": "ball", "robot_name": "robot"},
  )
  cfg.observations["critic"].terms["ball_vel_b"] = ObservationTermCfg(
    func=kick_mdp.ball_vel_b,
    params={"ball_name": "ball", "robot_name": "robot"},
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Domain-randomize ball mass (density) + friction.
  cfg.events["ball_mass"] = EventTermCfg(
    mode="reset",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("ball", body_names=("ball",)),
      "alpha_range": (-0.25, 0.25),  # ~e^±0.25 mass scale
    },
  )
  cfg.events["ball_friction"] = EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": (0.1, 0.8),
      "axes": [0],
    },
  )
  # Ball pose is set in KickMotionCommand after robot RSI (not here).

  # Boost kick-leg tracking so balance does not wash out the swing foot.
  _kick_leg = ("right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link")
  cfg.rewards["kick_foot_pos"] = RewardTermCfg(
    func=tracking_mdp.motion_relative_body_position_error_exp,
    weight=2.0,
    params={"command_name": "motion", "std": 0.15, "body_names": _kick_leg},
  )
  cfg.rewards["kick_foot_lin_vel"] = RewardTermCfg(
    func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
    weight=1.5,
    params={
      "command_name": "motion",
      "std": 0.8,
      "body_names": ("right_ankle_roll_link",),
    },
  )

  # Kick objective rewards (imitation rewards already in tracking base).
  cfg.rewards["ball_forward"] = RewardTermCfg(
    func=kick_mdp.ball_forward_progress,
    weight=5.0,
    params={"ball_name": "ball", "success_distance": 3.0},
  )
  cfg.rewards["kick_success"] = RewardTermCfg(
    func=kick_mdp.kick_success_bonus,
    weight=20.0,
    params={"ball_name": "ball", "success_distance": 3.0},
  )
  cfg.rewards["no_kick_timeout"] = RewardTermCfg(
    func=kick_mdp.no_kick_timeout_penalty,
    weight=-10.0,
    params={"ball_name": "ball", "min_forward": 0.3},
  )
  cfg.rewards["fall_penalty"] = RewardTermCfg(
    func=kick_mdp.robot_fallen,
    weight=-5.0,
    params={"min_height": 0.35},
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  cfg.terminations["ball_success"] = TerminationTermCfg(
    func=kick_mdp.ball_kicked_success,
    time_out=False,
    params={"ball_name": "ball", "success_distance": 3.0},
  )
  cfg.terminations["fallen"] = TerminationTermCfg(
    func=kick_mdp.robot_fallen,
    time_out=False,
    params={"min_height": 0.35},
  )

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, KickMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.loop = False
    motion_cmd.debug_vis = True

  return cfg
