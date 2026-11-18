import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import TerrainGeneratorCfg, flat, random_rough
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.velocity.mdp as mdp
import src.tasks.standing.mdp as standing_mdp
from src.tasks.standing.mdp.upper_body_motion import UpperBodyMotionCommandCfg

# Pose reward: legs/waist stay near default; arms stay loose (FALCON ref tracking).
_STANDING_POSE_STD_LEGS_WAIST = {
  r".*_hip_.*_joint": 0.05,
  r".*_knee_joint": 0.05,
  r".*_ankle_.*_joint": 0.05,
  r"waist_.*": 0.05,
}
_STANDING_POSE_STD_ARMS = {
  r".*shoulder.*": 1.0,
  r".*elbow.*": 1.0,
  r".*wrist.*": 1.0,
}


def make_standing_env_cfg() -> ManagerBasedRlEnvCfg:

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: one ObjRef per foot site.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "torso_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_torso_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.2
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "base_height_command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "base_height"},
    ),
    "ref_upper_dof_pos": ObservationTermCfg(
      func=standing_mdp.ref_upper_dof_pos,
      params={"command_name": "upper_body_motion"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "torso_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_torso_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "measured_base_height": ObservationTermCfg(
      func=mdp.base_height,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "terrain_sensor_name": "terrain_scan",
      },
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      nan_policy='sanitize'
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
      nan_policy='sanitize'
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      # Bias toward standing, but keep nonzero velocity for tracking robustness.
      rel_standing_envs=0.40,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      # Initial ranges match command_vel stage 0; later stages expand via curriculum.
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    ),
    "base_height": mdp.UniformBaseHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      default_height=0.95,
      rel_default_envs=0.3,
      ranges=mdp.UniformBaseHeightCommandCfg.Ranges(height=(0.45, 1.05)),
    ),
    "upper_body_motion": UpperBodyMotionCommandCfg(
      entity_name="robot",
      motion_dir="",
      resampling_time_range=(1.0e9, 1.0e9),
      fix_upper_body_prob=0.3,
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_standing_height": EventTermCfg(
      func=standing_mdp.ResetStandingHeightFromCommand,
      mode="reset",
      params={
        "height_command_name": "base_height",
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r".*_hip_pitch_joint",
            r".*_knee_joint",
            r".*_ankle_pitch_joint",
            r"waist_pitch_joint",
          ),
        ),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
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
    "torso_com_offset": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "scale",
        "ranges": (0.85, 1.15),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.50)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.50)},
    ),
    "track_base_height": RewardTermCfg(
      func=mdp.track_base_height,
      weight=1.0,
      params={
        "command_name": "base_height",
        "std": math.sqrt(0.01),
        "asset_cfg": SceneEntityCfg("robot"),  # Set per-robot body.
        "terrain_sensor_name": "terrain_scan",
      },
    ),
    "tracking_upper_body_dofs": RewardTermCfg(
      func=standing_mdp.tracking_upper_body_dofs,
      weight=1.0,
      params={
        "command_name": "upper_body_motion",
        "std": math.sqrt(0.3),
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=standing_mdp.DEFAULT_UPPER_ARM_JOINT_NAMES,
        ),
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.05),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=standing_mdp.variable_posture_height_aware,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "height_command_name": "base_height",
        "std_standing": {
          **_STANDING_POSE_STD_LEGS_WAIST,
          **_STANDING_POSE_STD_ARMS,
        },
        "std_walking": {
          **_STANDING_POSE_STD_LEGS_WAIST,
          **_STANDING_POSE_STD_ARMS,
        },
        "std_running": {
          **_STANDING_POSE_STD_LEGS_WAIST,
          **_STANDING_POSE_STD_ARMS,
        },
        "walking_threshold": 0.1,
        "running_threshold": 2.0,
      },
    ),
    "torso_ang_vel_l2": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "torso_angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/torso_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=standing_mdp.stand_still_height_gate,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "height_command_name": "base_height",
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r".*_hip_.*_joint",
            r".*_knee_joint",
            r".*_ankle_.*_joint",
            r"waist_.*",
          ),
        ),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
  }


  ##
  # Rough terrains (aligned with legged_gym: 8x8 m, 80% flat / 20% rough)
  ##
  rough_terrains_cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.8),
      "random_rough": random_rough(
        proportion=0.2,
        noise_range=(-0.05, 0.05),
        noise_step=0.005,
        downsampled_scale=0.2,
        horizontal_scale=0.1,
        vertical_scale=0.005,
      ),
    },
    add_lights=True,
  )


  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(rough_terrains_cfg),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
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
    episode_length_s=20.0,
  )