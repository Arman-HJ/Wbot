"""WBOT 20DoF suspended environment configuration.

Same hang and rewards as the 12DoF task, but the whole-body reference (legs +
arm swing) follows a shared gait schedule: every env tracks the same motion,
speed sweeps slow→fast in a loop and amplitude changes each loop.
"""

from dataclasses import replace
from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.suspended.mdp as mdp
from src.assets.robots import WBOT_20DOF_ACTION_SCALE, get_wbot_20dof_robot_cfg
from src.assets.robots.wbot.wbot_20dof_constants import get_spec
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES
from src.tasks.suspended.config.wbot_12dof.env_cfgs import (
  DECIMATION,
  SUSPEND_HEIGHT,
  WBOT_12DOF_JOINT_NAMES,
  suspended_spec,
  wbot_12dof_suspended_env_cfg,
)

GAIT_SCHEDULE = mdp.GaitScheduleCfg()

# Terms whose function reads the reference and therefore needs the schedule.
_SCHEDULED_FUNCS = {
  mdp.track_joint_pos,
  mdp.track_joint_vel,
  mdp.joint_pos_rmse,
  mdp.joint_vel_rmse,
  mdp.phase_observation,
  mdp.reference_joint_pos_rel,
}


def wbot_20dof_suspended_robot_cfg() -> EntityCfg:
  robot = get_wbot_20dof_robot_cfg(collision="feet", home_pose="stand")
  robot.spec_fn = partial(suspended_spec, get_spec)
  robot.init_state = replace(robot.init_state, pos=(0.0, 0.0, SUSPEND_HEIGHT))
  # Actuator lag is counted in physics steps: decimation=4 => 20 ms at 50 Hz.
  robot.articulation = replace(robot.articulation, actuators=tuple(
    replace(a, delay_min_lag=DECIMATION, delay_max_lag=DECIMATION)
    for a in robot.articulation.actuators
  ))
  return robot


def _to_20dof(value):
  if isinstance(value, SceneEntityCfg) and tuple(value.joint_names or ()) == WBOT_12DOF_JOINT_NAMES:
    return replace(value, joint_names=WBOT_JOINT_NAMES)
  return value


def wbot_20dof_suspended_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = wbot_12dof_suspended_env_cfg(play=play)
  term_groups = (
    *(group.terms.values() for group in cfg.observations.values()),
    cfg.rewards.values(),
    cfg.metrics.values(),
  )
  for terms in term_groups:
    for term in terms:
      for key, value in term.params.items():
        term.params[key] = _to_20dof(value)
      if term.func in _SCHEDULED_FUNCS:
        term.params["schedule"] = GAIT_SCHEDULE

  # Policy needs the current speed/amplitude to anticipate the reference.
  for group in cfg.observations.values():
    group.terms["gait_command"] = ObservationTermCfg(
      func=mdp.gait_command, params={"schedule": GAIT_SCHEDULE}
    )
  cfg.metrics["gait_freq_hz"] = MetricsTermCfg(
    func=mdp.gait_freq_hz, params={"schedule": GAIT_SCHEDULE}
  )
  cfg.metrics["gait_amp_scale"] = MetricsTermCfg(
    func=mdp.gait_amp_scale, params={"schedule": GAIT_SCHEDULE}
  )

  cfg.rewards["action_rate_l2"].weight = -1.0
  cfg.actions["joint_pos"].scale = WBOT_20DOF_ACTION_SCALE
  cfg.scene.entities = {"robot": wbot_20dof_suspended_robot_cfg()}
  return cfg
