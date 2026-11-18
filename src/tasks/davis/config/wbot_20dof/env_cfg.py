"""DAVIS goal-directed shooting env for wbot 20-DoF (mjlab)."""

from __future__ import annotations

import os
from typing import Literal

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.soccer.field import (
  FIELD_X,
  FIELD_Y,
  get_field_spec,
  get_goal_spec_posts,
)
from src.tasks.amp.mdp.observations import amp_state
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES
from src.tasks.davis import mdp as davis_mdp
from src.tasks.davis.mdp.depth import DepthPreprocessCfg
from src.tasks.davis.robot_wbot_neck import (
  DEPTH_RENDER_H,
  DEPTH_RENDER_W,
  HEAD_BODY_NAME,
  HEAD_CAM_POS,
  HEAD_CAM_QUAT_WXYZ,
  ZED2I_FOVY_DEG,
  davis_action_scale,
  get_wbot_davis_robot_cfg,
)

# Project page (#app-sim2real): calibrated ZED2i head-mounted depth at 168×94.
# Active 2-DoF neck on wbot (paper E1 has head yaw/pitch in the sensing loop).

# Scoring mouth (paper App. C-E): free-standing posts + crossbar.
GOAL_HALF_WIDTH = 1.20
GOAL_HEIGHT = 1.80
GOAL_DEPTH = 1.00
BALL_RADIUS = 0.11
BALL_MASS = 0.43  # mid of paper ball mass rand 0.36–0.56

# Pitch half-extents from football_jax (Lines.stl AABB).
# DAVIS: goal line at x=0 → playable x ∈ [0, 2*FIELD_X], y ∈ [-FIELD_Y, FIELD_Y].
assert FIELD_X == 7.0 and FIELD_Y == 4.5


def get_soccer_ball_spec(radius: float = BALL_RADIUS, mass: float = BALL_MASS) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(1.5, 0.0, radius))
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius,),
    mass=mass,
    rgba=(1.0, 0.55, 0.1, 1.0),
    friction=(0.8, 0.02, 0.01),
    condim=6,
  )
  return spec


def get_goal_spec(
  half_width: float = GOAL_HALF_WIDTH,
  height: float = GOAL_HEIGHT,
  depth: float = GOAL_DEPTH,
) -> mujoco.MjSpec:
  """2 posts + crossbar on the goal line (mouth +x). ``depth`` unused (API compat)."""
  del depth
  return get_goal_spec_posts(half_width=half_width, height=height)


def wbot_20dof_davis_shoot_env_cfg(
  play: bool = False,
  sanity: bool = False,
  pd_gains: Literal["armature_soft_ankle", "armature_soft_ankle_pace"] = "armature_soft_ankle",
) -> ManagerBasedRlEnvCfg:
  """Build DAVIS shooting env.

  ``sanity``: fewer envs, easier reset, shorter episode for smoke tests.
  """
  all_joints = SceneEntityCfg("robot")
  # AMP uses body 20-DoF only (paper: mask head/neck dims); football_jax-scale 4096 envs.
  amp_joints = SceneEntityCfg(
    "robot", joint_names=WBOT_JOINT_NAMES, preserve_order=True
  )
  amp_key_bodies = SceneEntityCfg(
    "robot", body_names=WBOT_KEY_BODY_NAMES, preserve_order=True
  )
  num_envs = 1 if play else (64 if sanity else 4096)

  depth_cfg = DepthPreprocessCfg()
  observations = {
    # Deployable proprio history (5 × 66). No ball XYZ.
    "actor_proprio": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=davis_mdp.observations.proprio_frame,
          params={"asset_cfg": all_joints},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=5,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    # Single-frame depth 168×80 after crop/normalize (+ DR in train).
    "actor_depth": ObservationGroupCfg(
      terms={
        "depth": ObservationTermCfg(
          func=davis_mdp.observations.depth_image,
          params={"sensor_name": "zed2i_depth", "training": not play},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=0,
      nan_policy="sanitize",
    ),
    # Train-only GT for annealing / aux loss. Actor model does NOT feed this raw to MLP.
    "aux_gt": ObservationGroupCfg(
      terms={
        "geo": ObservationTermCfg(
          func=davis_mdp.observations.aux_geometry_gt,
          params={
            "ball_name": "ball",
            "goal_name": "goal",
            "asset_cfg": all_joints,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=0,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=davis_mdp.observations.critic_frame,
          params={
            "asset_cfg": all_joints,
            "ball_name": "ball",
            "goal_name": "goal",
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
    # Discriminator state for DavisAmpPPO (not fed to actor/critic MLPs).
    "amp": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=amp_state,
          params={"joint_cfg": amp_joints, "key_body_cfg": amp_key_bodies},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=davis_action_scale(pd_gains),
      use_default_offset=True,
    )
  }

  events = {
    "reset_shoot": EventTermCfg(
      func=davis_mdp.events.reset_shoot_mixed,
      mode="reset",
      params={
        "penalty_prob": 0.7 if sanity else 0.5,
        "ball_name": "ball",
        "goal_name": "goal",
        "asset_cfg": all_joints,
      },
    ),
    # Project page #app-sim2real dynamics / camera DR.
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
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
    "ball_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("ball", body_names=("ball",)),
        "operation": "abs",
        "ranges": (0.36, 0.56),
      },
    ),
    "joint_offset": EventTermCfg(
      mode="startup",
      func=davis_mdp.events.randomize_joint_default_offset,
      params={"offset_range": (-0.02, 0.02)},
    ),
    "camera_pose": EventTermCfg(
      mode="startup",
      func=davis_mdp.events.randomize_camera_extrinsics,
      params={
        "sensor_name": "zed2i_depth",
        "pos_delta": 0.03,  # mid of 0.02–0.04 m
        "rot_delta": 0.04,  # mid of 0.025–0.060 rad
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 15.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
          "z": (-0.2, 0.2),
          "roll": (-0.78, 0.78),
          "pitch": (-0.78, 0.78),
          "yaw": (-0.78, 0.78),
        }
      },
    ),
    # Keep freejoint goal / field nailed to each env origin (multi-env + no tip).
    "pin_goal": EventTermCfg(
      func=davis_mdp.events.pin_goal,
      mode="interval",
      interval_range_s=(0.02, 0.02),
      params={"goal_name": "goal"},
    ),
    "pin_field": EventTermCfg(
      func=davis_mdp.events.pin_field,
      mode="interval",
      interval_range_s=(0.02, 0.02),
      params={"field_name": "field"},
    ),
  }

  # Chase-first then shoot. Contact terms kept but muted early via curriculum
  # ``contact_scale``; approach terms amplified via ``approach_scale``.
  _ankle_cfg = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    preserve_order=True,
  )
  rewards = {
    "approach_ball": RewardTermCfg(
      func=davis_mdp.rewards.approach_ball, weight=4.0, params={"ball_name": "ball"}
    ),
    "adaptive_velocity": RewardTermCfg(
      func=davis_mdp.rewards.adaptive_velocity, weight=5.0, params={"ball_name": "ball"}
    ),
    "progress_to_ball": RewardTermCfg(
      func=davis_mdp.rewards.progress_to_ball, weight=10.0, params={"ball_name": "ball"}
    ),
    "head_track_ball": RewardTermCfg(
      func=davis_mdp.rewards.head_track_ball, weight=1.0, params={"ball_name": "ball"}
    ),
    "keep_ball_in_horizon": RewardTermCfg(
      func=davis_mdp.rewards.keep_ball_in_horizon, weight=-0.5, params={"ball_name": "ball"}
    ),
    "body_follow_head": RewardTermCfg(
      func=davis_mdp.rewards.body_follow_head,
      weight=0.8,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "robot_behind_ball": RewardTermCfg(
      func=davis_mdp.rewards.robot_behind_ball,
      weight=1.5,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "lateral_kick_line": RewardTermCfg(
      func=davis_mdp.rewards.lateral_kick_line_penalty,
      weight=-1.5,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "head_yaw_align": RewardTermCfg(
      func=davis_mdp.rewards.head_yaw_align,
      weight=1.0,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "ankle_toward_ball": RewardTermCfg(
      func=davis_mdp.rewards.ankle_toward_ball,
      weight=1.5,
      params={"ball_name": "ball", "asset_cfg": _ankle_cfg},
    ),
    "kick_swing": RewardTermCfg(
      func=davis_mdp.rewards.kick_swing_to_goal,
      weight=4.0,
      params={"ball_name": "ball", "goal_name": "goal", "asset_cfg": _ankle_cfg},
    ),
    "ball_velocity": RewardTermCfg(
      func=davis_mdp.rewards.ball_velocity_to_goal,
      weight=8.0,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "ball_velocity_direction": RewardTermCfg(
      func=davis_mdp.rewards.ball_velocity_direction,
      weight=3.0,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "one_shot_score": RewardTermCfg(
      func=davis_mdp.rewards.one_shot_score,
      weight=40.0,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
    "stagnant_near_ball": RewardTermCfg(
      func=davis_mdp.rewards.stagnant_near_ball,
      weight=-2.0,
      params={"ball_name": "ball"},
    ),
    "action_rate": RewardTermCfg(func=davis_mdp.rewards.action_rate_l2, weight=-0.05),
    "joint_limits": RewardTermCfg(func=davis_mdp.rewards.joint_limits_l2, weight=-1.0),
    "flat_orientation": RewardTermCfg(func=davis_mdp.rewards.flat_orientation_l2, weight=-1.0),
    "lin_vel_z": RewardTermCfg(func=davis_mdp.rewards.lin_vel_z_l2, weight=-0.5),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fallen": TerminationTermCfg(
      func=davis_mdp.terminations.robot_fallen, params={"height_threshold": 0.35}
    ),
    "bad_ori": TerminationTermCfg(
      func=davis_mdp.terminations.bad_orientation, params={"limit_angle": 1.0}
    ),
    "ball_oob": TerminationTermCfg(
      # Field body + goal-mouth region are separate knobs (see terminations docstring).
      func=davis_mdp.terminations.ball_out_of_bounds,
      params={
        "field_x": float(FIELD_X),
        "field_y": float(FIELD_Y),
        "field_margin": 0.1,
        "goal_half_width": float(GOAL_HALF_WIDTH),
        "goal_depth": float(GOAL_DEPTH),
        "goal_line_margin": 0.05,
      },
    ),
    # End episode on goal so sparse score is not diluted by long dribble tails.
    "scored": TerminationTermCfg(
      func=davis_mdp.terminations.scored_success,
      params={"ball_name": "ball", "goal_name": "goal"},
    ),
  }

  curriculum = {
    "shoot": CurriculumTermCfg(
      func=davis_mdp.curriculums.shoot_curriculum,
      params={},
    )
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      # Full pitch is 2*FIELD_X ≈ 14 m; keep neighbour fields out of depth FOV.
      env_spacing=18.0,
      extent=3.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={} if play else curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=4.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    # Contact budgets aligned with football_jax wbot20dof (nconmax_per_env=48,
    # njmax=320) plus headroom for ball + fixed goal contacts under depth rendering.
    sim=SimulationCfg(
      nconmax=64,
      njmax=400,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,  # 200 Hz physics → 50 Hz policy (paper)
    episode_length_s=8.0 if sanity else 20.0,
  )

  cfg.scene.entities = {
    "robot": get_wbot_davis_robot_cfg(
      action_delay_range=(0, 0) if play else (2, 8),
      collision="full",
      pd_gains=pd_gains,
    ),
    "ball": EntityCfg(spec_fn=get_soccer_ball_spec),
    "goal": EntityCfg(spec_fn=get_goal_spec),
    "field": EntityCfg(spec_fn=get_field_spec),
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
      history_length=3,
      track_air_time=True,
    ),
    CameraSensorCfg(
      name="zed2i_depth",
      parent_body=f"robot/{HEAD_BODY_NAME}",
      pos=HEAD_CAM_POS,
      quat=HEAD_CAM_QUAT_WXYZ,
      fovy=ZED2I_FOVY_DEG,
      width=DEPTH_RENDER_W,
      height=DEPTH_RENDER_H,
      data_types=("depth",),
      use_textures=False,
      use_shadows=False,
      enabled_geom_groups=(0, 1, 2),
    ),
  )

  # Stash depth preprocess cfg for observation term.
  cfg._davis_depth_cfg = depth_cfg  # type: ignore[attr-defined]

  # Shoot-phase resume (DAVIS_SHOOT_PHASE=1): bias contact/score after chase is learned.
  if os.environ.get("DAVIS_SHOOT_PHASE", "").strip() in ("1", "true", "True", "yes"):
    shoot_boost = {
      "kick_swing": 8.0,
      "ball_velocity": 14.0,
      "ball_velocity_direction": 5.0,
      "one_shot_score": 60.0,
      "ankle_toward_ball": 2.5,
      "stagnant_near_ball": -4.0,
      # Keep chase, but don't let it drown kick once contact is on.
      "approach_ball": 3.0,
      "adaptive_velocity": 3.5,
      "progress_to_ball": 6.0,
    }
    for name, w in shoot_boost.items():
      if name in cfg.rewards:
        cfg.rewards[name].weight = float(w)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor_proprio"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("pd_gains", None)
    cfg.events.pop("ball_mass", None)
    cfg.events.pop("joint_offset", None)
    cfg.events.pop("camera_pose", None)
    # Play has no curriculum → lock spawn to stage-0 (narrowest) ranges.
    cfg.events["reset_shoot"].params.update(
      {
        "robot_dist_range": (2.0, 2.5),
        "robot_angle_range_deg": (-40.0, 40.0),
        "ball_x_range": (1.0, 2.0),
        "ball_y_range": (-1.0, 1.0),
      }
    )

  return cfg
