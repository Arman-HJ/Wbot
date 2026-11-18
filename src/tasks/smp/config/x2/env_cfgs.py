"""AgiBot X2 SMP environment configs."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef

from src.assets.robots import AGIBOT_X2_NO_HEAD_JOINT_EXPR, get_agibot_x2_smp_robot_cfg
from src.tasks.amp_v2 import mdp as amp_v2_mdp
from src.tasks.amp_v2.amp_v2_env_cfg import apply_uniform_reset_events
from src.tasks.amp_v2.config.x2.env_cfgs import X2_AMP_MOTION_BODY_NAMES
from src.tasks.amp_v2.config.x2.rl_cfg import DEFAULT_AMP_DATA_DIR
from src.tasks.amp_v2.mdp.commands import AmpExpertMotionCommandCfg, AmpExpertVisCommandCfg
from src.tasks.amp_v2.mdp.events import (
  init_amp_motion_reset_library,
  reset_robot_from_amp_motion,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.config.x2.env_cfgs import (
  agibot_x2_flat_env_cfg,
  agibot_x2_rough_env_cfg,
)

_DEFAULT_AMP_EXPERT_VIS_NPZ = DEFAULT_AMP_DATA_DIR / "locomotion_agibot_x2.npz"
_DEFAULT_AMP_RESET_DATASETS = {
  "locomotion_agibot_x2": 1.0,
  "locomotion_agibot_x2_sym": 1.0,
}

# Fixed command ranges (no command_vel curriculum).
_SMP_TWIST_RANGES = UniformVelocityCommandCfg.Ranges(
  lin_vel_x=(-1.0, 1.0),
  lin_vel_y=(-1.0, 1.0),
  ang_vel_z=(-1.0, 1.0),
)


def _apply_x2_smp_task_tweaks(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """SMP: no velocity curriculum, no gait reward, fixed (-1, 1) command ranges."""
  cfg.rewards.pop("foot_gait", None)

  twist = cfg.commands.get("twist")
  if isinstance(twist, UniformVelocityCommandCfg):
    twist.ranges.lin_vel_x = _SMP_TWIST_RANGES.lin_vel_x
    twist.ranges.lin_vel_y = _SMP_TWIST_RANGES.lin_vel_y
    twist.ranges.ang_vel_z = _SMP_TWIST_RANGES.ang_vel_z

  if play:
    return

  cfg.curriculum.pop("command_vel", None)


def _apply_x2_smp_infra(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Add SMP style-prior infrastructure on top of velocity v1 task env."""
  cfg.scene.entities = {"robot": get_agibot_x2_smp_robot_cfg()}

  no_head_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(AGIBOT_X2_NO_HEAD_JOINT_EXPR,),
  )

  imu_torso_ang_vel_cfg = BuiltinSensorCfg(
    name="imu_torso_ang_vel",
    sensor_type="frameangvel",
    obj=ObjRef(type="body", name="torso_link", entity="robot"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (imu_torso_ang_vel_cfg,)

  amp_v2_mdp.add_rsl_rl_amp_observation_groups(
    cfg,
    no_head_joint_cfg,
    ang_vel_sensor_name="robot/imu_torso_ang_vel",
    num_disc_obs_steps=10,
  )

  cfg.commands["motion"] = AmpExpertMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
    anchor_body_name="torso_link",
    body_names=X2_AMP_MOTION_BODY_NAMES,
  )
  cfg.commands["expert_ref"] = AmpExpertVisCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
    motion_file="",
    loop=True,
    anchor_root_to_robot=False,
    ref_char_offset=(0.0, 0.0, 0.0),
  )

  if play:
    apply_uniform_reset_events(cfg)
    twist = cfg.commands.get("twist")
    if isinstance(twist, UniformVelocityCommandCfg):
      twist.debug_vis = True
    expert_ref = cfg.commands.get("expert_ref")
    if isinstance(expert_ref, AmpExpertVisCommandCfg):
      expert_ref.motion_file = ""
      expert_ref.debug_vis = False
  else:
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events["init_amp_motion_library"] = EventTermCfg(
      func=init_amp_motion_reset_library,
      mode="startup",
      params={
        "amp_data_path": str(DEFAULT_AMP_DATA_DIR),
        "datasets": dict(_DEFAULT_AMP_RESET_DATASETS),
        "slow_down_factor": 1,
        "joint_asset_cfg": no_head_joint_cfg,
      },
    )
    cfg.events["reset_from_amp_motion"] = EventTermCfg(
      func=reset_robot_from_amp_motion,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_asset_cfg": no_head_joint_cfg,
        "pose_range": {
          "x": (-0.00, 0.00),
          "y": (-0.00, 0.00),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (-0.0, 0.0),
        },
        "velocity_range": {},
        "joint_position_range": (-0.05, 0.05),
        "joint_velocity_range": (-0.1, 0.1),
        "joint_position_scale_range": (1.0, 1.0),
        "joint_velocity_scale_range": (1.0, 1.0),
      },
    )
    cfg.events["add_joint_default_pos"] = EventTermCfg(
      mode="startup",
      func=dr.joint_default_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (-0.01, 0.01),
        "operation": "add",
      },
    )
    cfg.events["encoder_bias"] = EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    )
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
    expert_ref = cfg.commands.get("expert_ref")
    if isinstance(expert_ref, AmpExpertVisCommandCfg):
      if _DEFAULT_AMP_EXPERT_VIS_NPZ.is_file():
        expert_ref.motion_file = str(_DEFAULT_AMP_EXPERT_VIS_NPZ)
      expert_ref.debug_vis = True


def agibot_x2_smp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """SMP + velocity v1 task rewards on flat terrain."""
  cfg = agibot_x2_flat_env_cfg(play=play)
  _apply_x2_smp_infra(cfg, play=play)
  _apply_x2_smp_task_tweaks(cfg, play=play)
  return cfg


def agibot_x2_smp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """SMP + velocity v1 task rewards on rough terrain."""
  cfg = agibot_x2_rough_env_cfg(play=play)
  _apply_x2_smp_infra(cfg, play=play)
  _apply_x2_smp_task_tweaks(cfg, play=play)
  return cfg
