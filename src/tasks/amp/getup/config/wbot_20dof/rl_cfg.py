"""WBOT 20-DOF getup AMP runner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tasks.amp.locomotion.config.e1_21dof.rl_cfg import (
  AmpPpoAlgorithmCfg,
  E121DofWalkAmpRunnerCfg,
)
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES


_GETUP_FILES = (
  "src/assets/amp_motions/wbot_20dof/getup/cmu_getup/getup/111_06.npz",
  "src/assets/amp_motions/wbot_20dof/getup/cmu_getup/getup/111_07.npz",
  "src/assets/amp_motions/wbot_20dof/getup/cmu_getup/getup/111_08.npz",
)
_GETUP_LABELS = ("stand", "stand", "stand")


@dataclass
class Wbot20DofGetupAmpRunnerCfg(E121DofWalkAmpRunnerCfg):
  experiment_name: str = "wbot_20dof_getup_amp"
  algorithm: AmpPpoAlgorithmCfg = field(
    default_factory=lambda: AmpPpoAlgorithmCfg(
      amp_motion_files=_GETUP_FILES,
      amp_motion_weights=(1.0,) * len(_GETUP_FILES),
      amp_motion_labels=_GETUP_LABELS,
      amp_joint_names=WBOT_JOINT_NAMES,
      amp_key_body_names=WBOT_KEY_BODY_NAMES,
      amp_reward_coefficient=0.5,
      # Prioritize standing up first; style can catch up later.
      amp_task_reward_lerp=0.95,
      minimum_action_std=(0.05,) * 20,
      # Get-up poses are not left/right symmetric like locomotion.
      symmetry_cfg=None,
    )
  )


def wbot_20dof_getup_amp_runner_cfg() -> Wbot20DofGetupAmpRunnerCfg:
  return Wbot20DofGetupAmpRunnerCfg()
