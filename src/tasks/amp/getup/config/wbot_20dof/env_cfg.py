"""WBOT 20-DOF getup AMP task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots import get_wbot_20dof_robot_cfg
import src.tasks.amp.getup.mdp as getup_mdp
import src.tasks.amp.mdp as mdp
from src.tasks.amp.mdp import UniformVelocityCommandCfg
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES

def wbot_20dof_getup_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """WBOT getup AMP: CMU 111 getup experts + getup task rewards/terms.

  Task reward / success truncation match ``smp/.../getup_env_cfg.py`` (without the
  SMP guidance gate — AMP style reward replaces it). Heights are scaled to WBOT
  (standing head ≈ 0.89 m).
  """
  all_joints = SceneEntityCfg(
    "robot", joint_names=WBOT_JOINT_NAMES, preserve_order=True
  )
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  key_bodies = SceneEntityCfg(
    "robot", body_names=WBOT_KEY_BODY_NAMES, preserve_order=True
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.actor_frame,
          params={"joint_cfg": all_joints, "torso_cfg": torso},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.critic_frame,
          params={
            "joint_cfg": all_joints,
            "torso_cfg": torso,
            "feet_contact_sensor_name": "feet_ground_contact",
          },
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "amp": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=mdp.amp_state,
          params={"joint_cfg": all_joints, "key_body_cfg": key_bodies},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  # AMP runner still keys style labels off the twist command. Force standing so
  # every transition is labeled ``stand`` to match the getup expert clip.
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 5.0),
      rel_standing_envs=1.0,
      rel_turning_envs=0.0,
      rel_lateral_envs=0.0,
      turning_deadband=0.4,
      lateral_deadband=0.15,
      debug_vis=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        pure_turn_ang_vel_z=(0.0, 0.0),
        pure_lateral_lin_vel_y=(0.0, 0.0),
      ),
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (0.0, 0.0),
          "roll": (-math.pi, math.pi),
          "pitch": (-0.6 * math.pi, 0.6 * math.pi),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.3, 0.3),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.35, 0.35),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": all_joints,
      },
    ),
    "plant_on_ground": EventTermCfg(
      func=getup_mdp.plant_root_above_ground,
      mode="reset",
      params={"clearance": 0.01},
    ),
    "reset_stand_counter": EventTermCfg(
      func=getup_mdp.reset_stand_counter,
      mode="reset",
    ),
    "body_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": (-0.05, 0.05),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": torso,
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
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
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.0),
        "shared_random": True,
      },
    ),
  }

  # Task reward = getup_env_cfg's 0.7·upward + 0.3·head_height (no SMP product).
  rewards = {
    "task_getup": RewardTermCfg(
      func=getup_mdp.getup_task_reward,
      weight=8.0,
      params={
        "upward_weight": 0.7,
        "height_weight": 0.3,
        "target_velocity": 0.25,
        "head_height_threshold": 0.65,
        "upward_scale": 100.0,
        "target_height": 0.85,
        "height_scale": 1.0,
      },
    ),
    # Once the head is up, reward default pose (variable_posture standing-only).
    "pose": RewardTermCfg(
      func=getup_mdp.pose,
      weight=1.0,
      params={
        "asset_cfg": all_joints,
        "min_head_height": 0.80,
        "std": {
          r".*_hip_roll_joint": 0.08,
          r".*_hip_yaw_joint": 0.08,
          r".*_hip_pitch_joint": 0.12,
          r".*_knee_joint": 0.15,
          r".*_ankle_pitch_joint": 0.2,
          r".*_ankle_roll_joint": 0.2,
          r".*_shoulder_pitch_joint": 0.5,
          r".*_shoulder_roll_joint": 0.5,
          r".*_shoulder_yaw_joint": 0.5,
          r".*_elbow_joint": 0.5,
        },
      },
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.05),
    # Penalty. Curriculum raises |weight| after getup is learned.
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": all_joints}
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-50.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "stood_up": TerminationTermCfg(
      func=getup_mdp.stood_up,
      time_out=True,
      params={"head_height": 0.88, "max_speed": 0.5, "hold_steps": 25},
    ),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  # Initial penalty stays -0.01 so getup can be learned first.
  # From iter 3000, step every 500 iters: -0.05, -0.10, -0.5, -1.0.
  # One learning iteration = 24 control steps.
  curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 3000 * 24, "weight": -0.05},
          {"step": 3500 * 24, "weight": -0.10},
          {"step": 4000 * 24, "weight": -0.5},
          {"step": 4500 * 24, "weight": -1.0},
        ],
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1 if play else 4096,
      env_spacing=2.5,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=120,
      njmax=600,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=5.0,
  )
  cfg.scene.entities = {
    "robot": get_wbot_20dof_robot_cfg(
      action_delay_range=(0, 0) if play else (0, 2),
      with_head_site=True,
      enable_body_collision=True,
    )
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
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
      track_air_time=True,
    ),
  )

  if play:
    cfg.episode_length_s = int(1.0e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.clear()
    cfg.curriculum = {}

  return cfg
