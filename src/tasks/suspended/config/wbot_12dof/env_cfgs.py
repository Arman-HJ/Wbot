"""WBOT 12DoF suspended environment configuration."""

from collections.abc import Callable
from dataclasses import replace

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.robots import WBOT_12DOF_ACTION_SCALE, get_wbot_12dof_robot_cfg
from src.assets.robots.wbot.wbot_12dof_constants import get_spec
import src.tasks.suspended.mdp as mdp

WBOT_12DOF_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
SUSPEND_HEIGHT = 1.2
ANCHOR_HEIGHT = 2.0
# Shoulder tops on torso_link mesh: widest band |y|≈0.13 ends at z≈0.11.
SHOULDER_ATTACH_LOCAL = {
  "left": (0.0, 0.12, 0.11),
  "right": (0.0, -0.12, 0.11),
}
DECIMATION = 4


def wbot_12dof_joint_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WBOT_12DOF_JOINT_NAMES, preserve_order=True)


def suspended_spec(base_spec_fn: Callable[[], mujoco.MjSpec] = get_spec) -> mujoco.MjSpec:
  """Free-base robot hung from both shoulders by two unilateral spatial-tendon ropes.

  Anchors sit directly above the shoulder sites so both ropes are vertical and taut
  at rest; pelvis can still sway and pitch, so IMU readings are non-trivial.
  """
  spec = base_spec_fn()
  spec.body("pelvis").pos = [0.0, 0.0, SUSPEND_HEIGHT]
  torso = spec.body("torso_link")
  # Torso offset differs between XML revisions; assumes identity torso quat.
  tx, ty, tz = (float(v) for v in torso.pos)
  for side, local in SHOULDER_ATTACH_LOCAL.items():
    torso.add_site(
      name=f"hang_attach_{side}",
      pos=list(local),
      size=[0.015, 0.015, 0.015],
      rgba=[0.1, 0.9, 0.2, 1.0],
      group=0,
    )
    x, y, z = tx + local[0], ty + local[1], SUSPEND_HEIGHT + tz + local[2]
    spec.worldbody.add_site(
      name=f"hang_anchor_{side}",
      pos=[x, y, ANCHOR_HEIGHT],
      size=[0.025, 0.025, 0.025],
      rgba=[0.9, 0.2, 0.1, 1.0],
      group=0,
    )
    tendon = spec.add_tendon(
      name=f"hang_rope_{side}",
      limited=True,
      range=[0.0, ANCHOR_HEIGHT - z],
      width=0.01,
      rgba=[0.15, 0.15, 0.15, 1.0],
    )
    tendon.wrap_site(f"hang_anchor_{side}")
    tendon.wrap_site(f"hang_attach_{side}")
  return spec


def wbot_12dof_suspended_robot_cfg() -> EntityCfg:
  robot = get_wbot_12dof_robot_cfg()
  robot.spec_fn = suspended_spec
  robot.init_state = replace(robot.init_state, pos=(0.0, 0.0, SUSPEND_HEIGHT))
  # Actuator lag is counted in physics steps: decimation=4 => 20 ms at 50 Hz.
  robot.articulation = replace(robot.articulation, actuators=tuple(
    replace(a, delay_min_lag=DECIMATION, delay_max_lag=DECIMATION)
    for a in robot.articulation.actuators
  ))
  return robot


def wbot_12dof_suspended_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  del play  # Training and play share the same deterministic setup.
  joint_cfg = wbot_12dof_joint_asset_cfg()

  # Obs order: IMU first (base_ang_vel + projected_gravity), then joints / actions / phase / reference.
  # Deploy obs dim = 3+3+12+12+12+2+12 = 56.
  terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": joint_cfg}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": joint_cfg}),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(func=mdp.phase_observation),
    "reference_joint_pos": ObservationTermCfg(
      func=mdp.reference_joint_pos_rel, params={"asset_cfg": joint_cfg},
    ),
  }
  observations = {
    group: ObservationGroupCfg(
      terms=dict(terms), concatenate_terms=True, enable_corruption=False, history_length=1,
    )
    for group in ("actor", "critic")
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=WBOT_12DOF_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  events = {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "place_suspended_robot": EventTermCfg(func=mdp.place_suspended_robot, mode="reset"),
  }

  rewards = {
    "track_joint_position": RewardTermCfg(
      func=mdp.track_joint_pos, weight=2.0, params={"std": 0.05, "asset_cfg": joint_cfg},
    ),
    "track_joint_velocity": RewardTermCfg(
      func=mdp.track_joint_vel, weight=1.0, params={"std": 0.5, "asset_cfg": joint_cfg},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),
    "energy": RewardTermCfg(func=mdp.energy_cost, weight=-0.002, params={"asset_cfg": joint_cfg}),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "base_ang_vel_l2": RewardTermCfg(func=mdp.base_ang_vel_l2, weight=-0.05),
  }

  metrics = {
    "joint_position_rmse_rad": MetricsTermCfg(
      func=mdp.joint_pos_rmse, params={"asset_cfg": joint_cfg},
    ),
    "joint_velocity_rmse_rad_s": MetricsTermCfg(
      func=mdp.joint_vel_rmse, params={"asset_cfg": joint_cfg},
    ),
    "mechanical_power_w": MetricsTermCfg(func=mdp.energy_cost),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": wbot_12dof_suspended_robot_cfg()},
      sensors=(),
      num_envs=1,
      # Hang anchors are fixed world sites shared by every world, so all envs must
      # sit at the origin (worlds don't interact in mujoco_warp).
      env_spacing=0.0,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
      max_extra_envs=15,
    ),
    sim=SimulationCfg(
      # Rope tendon + free base needs a bit more constraint capacity than mocap hang.
      njmax=400,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    ),
    decimation=DECIMATION,
    episode_length_s=20.0,
  )
