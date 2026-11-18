"""AgiBot X2 velocity v2 environment configurations."""

import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  AGIBOT_X2_ACTION_SCALE,
  AGIBOT_X2_CFG,
)
from src.tasks.velocity_v2.velocity_env_cfg import (
  make_velocity_v2_env_cfg,
  validate_motion_command_cfg,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MOTION_FILE = os.environ.get(
  "X2_VELOCITY_V2_MOTION_FILE",
  str(_REPO_ROOT / "outputs" / "run_agibot_x2_full.npz"),
)

X2_MOTION_BODY_NAMES = (
  "pelvis",
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
  "waist_yaw_link",
  "waist_pitch_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "left_wrist_pitch_link",
  "left_wrist_roll_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
  "right_wrist_pitch_link",
  "right_wrist_roll_link",
  "head_yaw_link",
  "head_pitch_link",
)


def agibot_x2_velocity_v2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create X2 velocity-v2 task."""
  cfg = make_velocity_v2_env_cfg()
  cfg.scene.entities = {"robot": AGIBOT_X2_CFG}

  imu_ang_vel_cfg = BuiltinSensorCfg(
    name="imu_ang_vel",
    sensor_type="frameangvel",
    obj=ObjRef(type="body", name="pelvis", entity="robot"),
  )
  imu_lin_vel_cfg = BuiltinSensorCfg(
    name="imu_lin_vel",
    sensor_type="framelinvel",
    obj=ObjRef(type="body", name="pelvis", entity="robot"),
  )
  root_angmom_cfg = BuiltinSensorCfg(
    name="root_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name="pelvis", entity="robot"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    imu_ang_vel_cfg,
    imu_lin_vel_cfg,
    root_angmom_cfg,
  )
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "scale",
      "ranges": (0.85, 1.15),
    },
  )
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(4.0, 6.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "roll": (-0.35, 0.35),
        "pitch": (-0.35, 0.35),
        "yaw": (-0.5, 0.5),
      }
    },
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.entity_name = "robot"
  motion_cmd.motion_file = DEFAULT_MOTION_FILE
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = X2_MOTION_BODY_NAMES
  motion_cmd.resampling_time_range = (1.0e9, 1.0e9)
  motion_cmd.debug_vis = True
  motion_cmd.pose_range = {} if play else {
    "x": (-0.03, 0.03),
    "y": (-0.03, 0.03),
    "z": (-0.01, 0.01),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.10, 0.10),
  }
  motion_cmd.velocity_range = {} if play else {
    "x": (-0.2, 0.2),
    "y": (-0.1, 0.1),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  motion_cmd.joint_position_range = (0.0, 0.0) if play else (-0.05, 0.05)
  motion_cmd.sampling_mode = "start" if play else "adaptive"
  motion_cmd.start_time_step = 0
  motion_cmd.start_time_step_prob = 0.2 if not play else 0.0

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("phase", None)
    cfg.observations[group_name].terms.pop("height_scan", None)

  cfg.rewards.pop("foot_gait", None)
  cfg.rewards.pop("pose", None)
  cfg.rewards.pop("stand_still", None)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = AGIBOT_X2_ACTION_SCALE

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.5, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = None

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    cfg.curriculum = {}

  validate_motion_command_cfg(cfg)
  return cfg
