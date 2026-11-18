"""Decoupled velocity task configuration (dual-agent, following FALCON).

The decoupled task splits a full-body velocity policy into two independently
trained PPO agents:

- ``lower_body``: leg joints (12 DoF) — gait / locomotion tracking.
- ``upper_body``: torso + arm joints (17 DoF) — stability & angular-momentum
  balance while the legs drive gait.

Observations are intentionally NOT split: both agents share the same ``actor`` /
``critic`` observation groups (only actions and rewards are decoupled).

Robot-specific configurations (X2, Adam SP, ...) call
:func:`make_decoupled_velocity_env_cfg` and customize as needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity_v3.velocity_env_cfg import make_velocity_v3_env_cfg
from src.tasks.velocity_decoupled.mdp import FixedBaseHeightCommandCfg
import src.tasks.velocity.mdp as mdp


@dataclass
class DecoupledRobotSpec:
  """Robot-specific names required by the decoupled task."""

  # Action split (actuator/joint name regex expressions).
  lower_body_joint_expr: tuple[str, ...]
  upper_body_joint_expr: tuple[str, ...]
  # Feet / torso for sensors, rewards, and viewer.
  foot_sole_sites: tuple[str, ...] = ("left_foot", "right_foot")
  ankle_link_pattern: str = r"^(left_ankle_roll_link|right_ankle_roll_link)$"
  torso_body: str = "torso_link"
  foot_collision_geoms: tuple[str, ...] = ()
  # Target base height above ground (m).
  target_base_height: float = 0.66
  # Sensor names (must exist on the robot's sensors or be created by the caller).
  angvel_sensor: str = "robot/imu_torso_ang_vel"
  linvel_sensor: str = "robot/imu_torso_lin_vel"
  angmom_sensor: str = "robot/torso_angmom"
  gravity_sensor: str = "robot/imu_upvector"
  # Self-collision contact pattern (subtree).
  self_collision_pattern: str = "pelvis"
  # IMU site name used by the startup IMU orientation DR event.
  imu_site: str = "imu"
  # Height of torso used by base_height reward / upright reward.
  base_height_body: str = "torso_link"
  # Action scale (float or dict), defaults to the base v3 joint_pos scale.
  action_scale: float | dict = field(default_factory=float)
  # Whether the caller creates torso IMU / angmom sensors in the scene. Robots whose
  # assets already carry these sensors (e.g. Adam SP) set this to False.
  create_torso_sensors: bool = True


def make_decoupled_velocity_env_cfg(
  robot_cfg: ManagerBasedRlEnvCfg.scene.entities,  # type: ignore[name-defined]
  spec: DecoupledRobotSpec,
  *,
  play: bool = False,
  terrain: str = "flat",
) -> ManagerBasedRlEnvCfg:
  """Build a decoupled velocity env config for the given robot spec.

  Args:
    robot_cfg: robot EntityCfg to place in the scene as ``robot``.
    spec: robot-specific joint/body/site/sensor names.
    play: play-mode overrides (infinite episode, fixed command).
    terrain: ``"flat"`` or ``"rough"``.
  """
  cfg = make_velocity_v3_env_cfg()
  cfg.scene.entities = {"robot": robot_cfg}

  site_names = spec.foot_sole_sites
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=spec.ankle_link_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=spec.self_collision_pattern, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=spec.self_collision_pattern, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )
  if spec.create_torso_sensors:
    imu_torso_ang_vel_cfg = BuiltinSensorCfg(
      name="imu_torso_ang_vel",
      sensor_type="frameangvel",
      obj=ObjRef(type="body", name=spec.torso_body, entity="robot"),
    )
    imu_lin_vel_cfg = BuiltinSensorCfg(
      name="imu_torso_lin_vel",
      sensor_type="framelinvel",
      obj=ObjRef(type="body", name=spec.torso_body, entity="robot"),
    )
    torso_angmom_cfg = BuiltinSensorCfg(
      name="torso_angmom",
      sensor_type="subtreeangmom",
      obj=ObjRef(type="body", name=spec.torso_body, entity="robot"),
    )
    cfg.scene.sensors = cfg.scene.sensors + (
      imu_torso_ang_vel_cfg,
      imu_lin_vel_cfg,
      torso_angmom_cfg,
    )

  # ---- Actions: split into lower (legs) and upper (torso+arms) blocks. ----
  joint_scale = cfg.actions["joint_pos"].scale
  if not isinstance(spec.action_scale, float):
    joint_scale = spec.action_scale
  cfg.actions = {
    "joint_pos_lower": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=spec.lower_body_joint_expr,
      scale=joint_scale,
      use_default_offset=True,
    ),
    "joint_pos_upper": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=spec.upper_body_joint_expr,
      scale=joint_scale,
      use_default_offset=True,
    ),
  }

  # ---- Commands: fixed base-height command added to the twist command. ----
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.viz.z_offset = 1.50
  cfg.commands["base_height"] = FixedBaseHeightCommandCfg(
    entity_name="robot",
    target_height=spec.target_base_height,
  )

  # ---- Obs additions: base-height command + torso angular momentum. ----
  # Fix robot-specific sensor names in the base v3 obs (X2 names differ from
  # assets that carry their own sensors, e.g. Adam SP).
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    if "torso_ang_vel" in terms:
      terms["torso_ang_vel"].params["sensor_name"] = spec.angvel_sensor
    if "torso_lin_vel" in terms:
      terms["torso_lin_vel"].params["sensor_name"] = spec.linvel_sensor
    if "projected_gravity" in terms:
      terms["projected_gravity"].params["sensor_name"] = spec.gravity_sensor
  cfg.observations["actor"].terms["command_base_height"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "base_height"},
  )
  cfg.observations["actor"].terms["torso_angmom"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": spec.angmom_sensor},
  )
  cfg.observations["critic"].terms["command_base_height"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "base_height"},
  )
  cfg.observations["critic"].terms["torso_angmom"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": spec.angmom_sensor},
  )

  # ---- Rewards: split into lower (gait) and upper (stability) groups. ----
  cfg.rewards = _decoupled_rewards(spec)

  # ---- Terminations / events / curriculum. ----
  if spec.foot_collision_geoms:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = spec.foot_collision_geoms
  cfg.events["torso_com_offset"].params["asset_cfg"].body_names = (spec.torso_body,)
  # IMU site name is robot-specific (X2: imu, Adam SP: imu_in_pelvis).
  cfg.events["imu_site_quat"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=(spec.imu_site,)
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1,
    group="lower_body",
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
        {"step": 20000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
        {"step": 40000 * 24, "lin_vel_x": (-1.5, 3.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
      ],
    },
  )

  # ---- Terrain: flat removes terrain scan; rough keeps it. ----
  assert cfg.scene.terrain is not None
  if terrain == "flat":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)
    cfg.scene.sensors = tuple(
      s for s in (cfg.scene.sensors or ()) if getattr(s, "name", None) != "terrain_scan"
    )
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)

  # ---- Play-mode overrides. ----
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.viewer.body_name = spec.torso_body

  return cfg


def _decoupled_rewards(spec: DecoupledRobotSpec) -> dict[str, RewardTermCfg]:
  """Reward groups: ``lower_body`` (gait/velocity) and ``upper_body`` (stability)."""
  rewards: dict[str, RewardTermCfg] = {}

  # ---- Lower-body group: locomotion / gait. ----
  rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity,
    weight=2.0,
    group="lower_body",
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity,
    weight=2.0,
    group="lower_body",
    params={"command_name": "twist", "std": math.sqrt(0.50)},
  )
  rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_exp,
    weight=1.0,
    group="lower_body",
    params={
      "target_height": spec.target_base_height,
      "std": math.sqrt(0.01),
      "asset_cfg": SceneEntityCfg("robot", body_names=(spec.base_height_body,)),
    },
  )
  rewards["feet_gait"] = RewardTermCfg(
    func=mdp.feet_gait,
    weight=1.0,
    group="lower_body",
    params={
      "period": 0.8,
      "offset": [0.0, 0.5],
      "threshold": 0.56,
      "command_threshold": 0.1,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
    },
  )
  rewards["feet_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance,
    weight=-10.0,
    group="lower_body",
    params={
      "target_height": 0.10,
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=spec.foot_sole_sites),
    },
  )
  rewards["feet_slip"] = RewardTermCfg(
    func=mdp.feet_slip,
    weight=-0.25,
    group="lower_body",
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=spec.foot_sole_sites),
    },
  )
  rewards["soft_landing"] = RewardTermCfg(
    func=mdp.soft_landing,
    weight=-1e-3,
    group="lower_body",
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  # ---- Shared penalties: applied to BOTH lower and upper agents. ----
  # The decoupled reward manager aggregates each term into exactly one group, so
  # whole-body penalties (joint acc / limits / action rate / stand still /
  # termination) are duplicated once per agent group.
  for _suffix, _group in (("_lower", "lower_body"), ("_upper", "upper_body")):
    rewards[f"joint_acc_l2{_suffix}"] = RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-2.5e-7, group=_group
    )
    rewards[f"joint_pos_limits{_suffix}"] = RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-10.0, group=_group
    )
    rewards[f"action_rate_l2{_suffix}"] = RewardTermCfg(
      func=mdp.action_rate_l2, weight=-0.05, group=_group
    )
    rewards[f"stand_still{_suffix}"] = RewardTermCfg(
      func=mdp.stand_still,
      weight=-2.0,
      group=_group,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    )
    rewards[f"is_terminated{_suffix}"] = RewardTermCfg(
      func=mdp.is_terminated, weight=-200.0, group=_group
    )

  # ---- Upper-body group: stability & angular-momentum balance. ----
  rewards["upright"] = RewardTermCfg(
    func=mdp.upright,
    weight=1.0,
    group="upper_body",
    params={
      "std": math.sqrt(0.05),
      "gravity_sensor_name": spec.gravity_sensor,
      "asset_cfg": SceneEntityCfg("robot", body_names=(spec.torso_body,)),
    },
  )
  rewards["torso_ang_vel_l2"] = RewardTermCfg(
    func=mdp.body_angular_velocity_penalty,
    weight=-0.05,
    group="upper_body",
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(spec.torso_body,))},
  )
  rewards["torso_angular_momentum"] = RewardTermCfg(
    func=mdp.angular_momentum_penalty,
    weight=-0.025,
    group="upper_body",
    params={"sensor_name": spec.angmom_sensor},
  )
  rewards["upper_body_pose"] = RewardTermCfg(
    func=mdp.variable_posture,
    weight=0.5,
    group="upper_body",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=spec.upper_body_joint_expr),
      "command_name": "twist",
      "std_standing": {".*": 0.05},
      "std_walking": {
        r".*waist.*": 0.05,
        r".*shoulder.*": 0.15,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.15,
      },
      "std_running": {
        r".*waist.*": 0.1,
        r".*shoulder.*": 0.25,
        r".*elbow.*": 0.2,
        r".*wrist.*": 0.15,
      },
      "walking_threshold": 0.1,
      "running_threshold": 2.0,
    },
  )

  return rewards
