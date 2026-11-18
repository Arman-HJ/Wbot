"""WBOT 12-DOF (leg-only) flat-ground velocity-commanded locomotion AMP task.

Same task as the 20-DOF walk/run AMP (commands, rewards, curriculum), on the
leg-only model. The 12-DOF legs and total mass match the 20-DOF robot; the arms
are welded to the torso.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.wbot.wbot_12dof_constants import get_spec
from src.assets.robots.wbot.wbot_20dof_constants import (
  WBOT_20DOF_ARTICULATION,
  WBOT_20DOF_FOOT_COLLISION,
  WBOT_20DOF_HOME_STAND,
)
from src.tasks.amp.locomotion.config.wbot_20dof.env_cfg import (
  wbot_20dof_walk_amp_env_cfg,
)
from src.tasks.amp.wbot_constants import (
  WBOT_12DOF_JOINT_NAMES,
  WBOT_12DOF_KEY_BODY_NAMES,
  WBOT_JOINT_NAMES,
  WBOT_KEY_BODY_NAMES,
)

_ARM_REWARDS = ("stand_shoulder_roll", "stand_shoulder_yaw")


def get_wbot_12dof_amp_robot_cfg(
  action_delay_range: tuple[int, int] = (0, 0),
) -> EntityCfg:
  delay_min_lag, delay_max_lag = action_delay_range
  actuators = tuple(
    deepcopy(act)
    for act in WBOT_20DOF_ARTICULATION.actuators
    if not any("shoulder" in e or "elbow" in e for e in act.target_names_expr)
  )
  for actuator in actuators:
    actuator.delay_min_lag = delay_min_lag
    actuator.delay_max_lag = delay_max_lag
    actuator.delay_hold_prob = 0.9 if delay_max_lag > 0 else 0.0
    actuator.delay_update_period = 4 if delay_max_lag > 0 else 0

  init_state = deepcopy(WBOT_20DOF_HOME_STAND)
  init_state.joint_pos = {
    name: value
    for name, value in init_state.joint_pos.items()
    if name in WBOT_12DOF_JOINT_NAMES
  }
  return EntityCfg(
    init_state=init_state,
    collisions=(WBOT_20DOF_FOOT_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=WBOT_20DOF_ARTICULATION.soft_joint_pos_limit_factor,
    ),
  )


def _to_12dof(entity_cfg: SceneEntityCfg) -> SceneEntityCfg:
  if tuple(entity_cfg.joint_names or ()) == WBOT_JOINT_NAMES:
    return replace(entity_cfg, joint_names=WBOT_12DOF_JOINT_NAMES)
  if tuple(entity_cfg.body_names or ()) == WBOT_KEY_BODY_NAMES:
    return replace(entity_cfg, body_names=WBOT_12DOF_KEY_BODY_NAMES)
  return entity_cfg


def wbot_12dof_walk_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the WBOT 12-DOF walk-and-run AMP task."""
  cfg = wbot_20dof_walk_amp_env_cfg(play=play)

  for name in _ARM_REWARDS:
    cfg.rewards.pop(name)
  # 12DoF symmetry (data_augmentation_func_12dof) expects the base critic frame.
  cfg.observations["critic"].terms["frame"].params.pop("landing_force_scale", None)

  term_groups = (
    *(group.terms.values() for group in cfg.observations.values()),
    cfg.rewards.values(),
    cfg.events.values(),
    cfg.terminations.values(),
  )
  for terms in term_groups:
    for term in terms:
      for key, value in term.params.items():
        if isinstance(value, SceneEntityCfg):
          term.params[key] = _to_12dof(value)

  cfg.scene.entities = {
    "robot": get_wbot_12dof_amp_robot_cfg(
      action_delay_range=(0, 0) if play else (0, 2)
    )
  }
  return cfg
