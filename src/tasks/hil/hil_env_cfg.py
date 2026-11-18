"""HIL base environment configuration (velocity tracking task, no heading)."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.amp_v2.mdp as amp_mdp
import src.tasks.hil.mdp as hil_mdp
import src.tasks.velocity.mdp as mdp
from src.tasks.amp_v2.mdp.commands import AmpExpertMotionCommandCfg
from src.tasks.hil.mdp.twist_command import HilTwistCommandCfg


def make_hil_env_cfg() -> ManagerBasedRlEnvCfg:
  """HIL hybrid env: 50% motion tracking + 50% AMP velocity task."""

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "task_indicator": ObservationTermCfg(func=hil_mdp.task_indicator),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=amp_mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=amp_mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=amp_mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=amp_mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      nan_policy="sanitize",
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": HilTwistCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 7.0),
      debug_vis=False,
      ranges=HilTwistCommandCfg.Ranges(
        lin_vel_x=(0.5, 3.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
      ),
    ),
    "motion": AmpExpertMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
      anchor_body_name="",
      body_names=(),
    ),
    # Expert ghost for training videos / play (--amp-vis-motion or env_cfgs).
    "expert_ref": amp_mdp.AmpExpertVisCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
      motion_file="",
      loop=True,
      anchor_root_to_robot=False,
      ref_char_offset=(0.0, 0.0, 0.0),
    ),
  }

  events = {
    "init_hil_training": EventTermCfg(
      func=hil_mdp.init_hil_training,
      mode="startup",
      params={
        "track_fraction": 0.5,
        "amp_data_path": "",
        "datasets": {},
        "slow_down_factor": 1,
        "joint_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_hil_from_amp_motion": EventTermCfg(
      func=hil_mdp.reset_hil_from_amp_motion,
      mode="reset",
      params={
        "track_fraction": 0.5,
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "pose_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
        },
        "velocity_range": {},
        "joint_position_range": (0.0, 0.0),
        "psi_pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (-0.05, 0.05),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-0.2, 0.2),
        },
        "psi_velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.1, 0.1),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.3, 0.3),
        },
        "psi_joint_position_range": (-0.05, 0.05),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  vel_std = math.sqrt(0.25)
  rewards = {
    "track_anchor_pos": RewardTermCfg(
      func=hil_mdp.hil_track_anchor_pos_reward,
      weight=0.15,
      params={"command_name": "motion", "std": 0.3},
    ),
    "track_anchor_rot": RewardTermCfg(
      func=hil_mdp.hil_track_anchor_rot_reward,
      weight=0.15,
      params={"command_name": "motion", "std": 0.4},
    ),
    "track_root_lin_vel": RewardTermCfg(
      func=hil_mdp.hil_track_root_lin_vel_reward,
      weight=0.1,
      params={"std": 1.0},
    ),
    "track_root_ang_vel": RewardTermCfg(
      func=hil_mdp.hil_track_root_ang_vel_reward,
      weight=0.1,
      params={"std": 3.14},
    ),
    "amp_track_linear_velocity": RewardTermCfg(
      func=hil_mdp.hil_amp_track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": vel_std},
    ),
    "amp_track_angular_velocity": RewardTermCfg(
      func=hil_mdp.hil_amp_track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": vel_std},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "track_bad_joint_pos": TerminationTermCfg(
      func=hil_mdp.hil_track_bad_joint_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "threshold": 0.5,
      },
    ),
    "amp_fell_over": TerminationTermCfg(
      func=hil_mdp.hil_amp_fell_over,
      params={"limit_angle": math.radians(70), "asset_cfg": SceneEntityCfg("robot")},
    ),
    "amp_root_height_below_minimum": TerminationTermCfg(
      func=hil_mdp.hil_amp_root_height_below_minimum,
      params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
      ),
      sensors=(),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.25,
        use_default_offset=True,
      )
    },
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={
      "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )


__all__ = ["make_hil_env_cfg"]
