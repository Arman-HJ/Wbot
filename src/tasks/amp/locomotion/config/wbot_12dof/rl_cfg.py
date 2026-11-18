"""WBOT 12-DOF locomotion AMP runner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tasks.amp.locomotion.config.e1_21dof.rl_cfg import (
  AmpPpoAlgorithmCfg,
  E121DofWalkAmpRunnerCfg,
)
from src.tasks.amp.wbot_constants import (
  WBOT_12DOF_JOINT_NAMES,
  WBOT_12DOF_KEY_BODY_NAMES,
)


# Leg slices of the WBOT 20-DOF clips (scripts/make_wbot_12dof_amp_dataset.py).
_FILES = (
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/walk_moving.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/run.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/run_mirror.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/turn_l.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/turn_r.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/side_l.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/side_r.npz",
  "src/assets/amp_motions/wbot_12dof/walk_run_amp/stand.npz",
)
_LABELS = (
  "walk",
  "run",
  "run",
  "turn",
  "turn",
  "side",
  "side",
  "stand",
)


@dataclass
class Wbot12DofWalkAmpRunnerCfg(E121DofWalkAmpRunnerCfg):
  experiment_name: str = "wbot_12dof_walk_run_amp"
  algorithm: AmpPpoAlgorithmCfg = field(
    default_factory=lambda: AmpPpoAlgorithmCfg(
      amp_motion_files=_FILES,
      amp_motion_weights=(1.0,) * len(_FILES),
      amp_motion_labels=_LABELS,
      amp_joint_names=WBOT_12DOF_JOINT_NAMES,
      amp_key_body_names=WBOT_12DOF_KEY_BODY_NAMES,
      minimum_action_std=(0.05,) * len(WBOT_12DOF_JOINT_NAMES),
      symmetry_cfg={
        "use_data_augmentation": True,
        "data_augmentation_func": "src.tasks.amp.symmetry_wbot:data_augmentation_func_12dof",
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
      },
    )
  )


def wbot_12dof_walk_amp_runner_cfg() -> Wbot12DofWalkAmpRunnerCfg:
  return Wbot12DofWalkAmpRunnerCfg()
