"""WBOT 20-DOF locomotion AMP runner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tasks.amp.locomotion.config.e1_21dof.rl_cfg import (
  AmpPpoAlgorithmCfg,
  E121DofWalkAmpRunnerCfg,
)
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES


# E1 AMP clips retargeted onto WBOT with GMR. Order matches CLIPS.
# Canonical pack: src/assets/amp_motions/wbot_20dof/walk_run_amp/ (videos in video/).
# walk_straight_* = mid-speed straight segments cut from walk_moving, yaw-drift
# removed and mirrored by scripts/export_straight_walk_clips.py.
_FILES = (
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/walk_straight_a.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/walk_straight_a_mirror.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/walk_straight_b.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/walk_straight_b_mirror.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/run.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/run_mirror.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/turn_l.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/turn_r.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/side_l.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/side_r.npz",
  "src/assets/amp_motions/wbot_20dof/walk_run_amp/stand.npz",
)
_LABELS = (
  "walk",
  "walk",
  "walk",
  "walk",
  "run",
  "run",
  "turn",
  "turn",
  "side",
  "side",
  "stand",
)
# Each straight segment is half a clip, so a segment plus its mirror keeps the
# previous walk weight of 1.0. Run/turn/side/stand stay at 1.0.
_WEIGHTS = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class Wbot20DofWalkAmpRunnerCfg(E121DofWalkAmpRunnerCfg):
  experiment_name: str = "wbot_20dof_walk_run_amp"
  algorithm: AmpPpoAlgorithmCfg = field(
    default_factory=lambda: AmpPpoAlgorithmCfg(
      amp_motion_files=_FILES,
      amp_motion_weights=_WEIGHTS,
      amp_motion_labels=_LABELS,
      amp_joint_names=WBOT_JOINT_NAMES,
      amp_key_body_names=WBOT_KEY_BODY_NAMES,
      amp_discriminator_type="part_gate",
      amp_part_gate_dim=128,
      amp_standing_command_threshold=0.05,
      minimum_action_std=(0.05,) * 20,
      symmetry_cfg={
        "use_data_augmentation": True,
        "data_augmentation_func": (
          "src.tasks.amp.symmetry_wbot:data_augmentation_func"
        ),
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
      },
    )
  )


def wbot_20dof_walk_amp_runner_cfg() -> Wbot20DofWalkAmpRunnerCfg:
  return Wbot20DofWalkAmpRunnerCfg()
