"""Observation groups for the long-history encoder actor (football / HumanoidKick style).

- ``actor_history``: ``HISTORY_LENGTH`` policy steps of proprioception, time-major
  oldest → newest, copies of the matching ``actor`` terms (same noise).
- ``privileged``: decoder target (training-only, never deployed): states the
  actor no longer observes + sampled DR parameters.
"""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.tracking import mdp as tracking_mdp

HISTORY_LENGTH = 50
HISTORY_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


def add_history_encoder_observations(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_geoms: str,
  play: bool = False,
  exclude_privileged: tuple[str, ...] = (),
) -> None:
  """Add ``actor_history`` and ``privileged`` groups; call after the actor group is final.

  ``exclude_privileged`` drops terms whose DR is not enabled in the task (they
  would be constant targets).
  """
  actor_terms = cfg.observations["actor"].terms
  # Group-level history_length mutates term cfgs, so the history group gets copies.
  cfg.observations["actor_history"] = ObservationGroupCfg(
    terms={k: copy.deepcopy(actor_terms[k]) for k in HISTORY_TERMS},
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=HISTORY_LENGTH,
    history_ordering="time",
  )
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  privileged_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=tracking_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "anchor_height": ObservationTermCfg(
      func=tracking_mdp.robot_anchor_height, params={"command_name": "motion"}
    ),
    "encoder_bias": ObservationTermCfg(func=tracking_mdp.joint_encoder_bias),
    "pd_gain_scale": ObservationTermCfg(func=tracking_mdp.pd_gain_scale),
    "torso_mass_scale": ObservationTermCfg(
      func=tracking_mdp.body_mass_scale, params={"asset_cfg": torso}
    ),
    "torso_com_offset": ObservationTermCfg(
      func=tracking_mdp.body_com_offset,
      params={"asset_cfg": copy.deepcopy(torso)},
    ),
    "foot_friction": ObservationTermCfg(
      func=tracking_mdp.geom_friction_coef,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=foot_geoms)},
    ),
  }
  unknown = set(exclude_privileged) - set(privileged_terms)
  if unknown:
    raise ValueError(f"Unknown privileged terms to exclude: {sorted(unknown)}")
  cfg.observations["privileged"] = ObservationGroupCfg(
    terms={k: v for k, v in privileged_terms.items() if k not in exclude_privileged},
    concatenate_terms=True,
    enable_corruption=False,
  )
