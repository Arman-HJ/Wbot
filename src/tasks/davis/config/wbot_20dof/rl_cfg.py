"""RSL-RL config for DAVIS shooting on wbot 20-DoF (+ AMP from HumanoidSoccer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlModelCfg

from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SOCCER_AMP_DIR = "src/assets/amp_motions/wbot_20dof/soccer_standard"
_WALK_AMP_DIR = "src/assets/amp_motions/wbot_20dof/walk_run_amp"
_SOCCER_FILES = tuple(
  f"{_SOCCER_AMP_DIR}/soccer-standard-{i:03d}_{side}.npz"
  for i, side in (
    (1, "right"),
    (2, "left"),
    (3, "left"),
    (4, "right"),
    (5, "right"),
    (6, "right"),
    (7, "left"),
    (8, "left"),
    (9, "right"),
    (10, "right"),
  )
)
# Straight walk (+ mirrors) from AMP-Walk-Run; same walk bucket as soccer.
_WALK_FILES = (
  f"{_WALK_AMP_DIR}/walk_straight_a.npz",
  f"{_WALK_AMP_DIR}/walk_straight_a_mirror.npz",
  f"{_WALK_AMP_DIR}/walk_straight_b.npz",
  f"{_WALK_AMP_DIR}/walk_straight_b_mirror.npz",
)
# Single style bucket (DavisAmpPPO.FIXED_AMP_LABEL == "walk").
# Soccer keeps weight 1.0; walk_straight 0.5 each (segment+mirror ≈ 1.0 per gait).
_AMP_FILES = _SOCCER_FILES + _WALK_FILES
_AMP_LABELS = ("walk",) * len(_AMP_FILES)
_AMP_WEIGHTS = (1.0,) * len(_SOCCER_FILES) + (0.5,) * len(_WALK_FILES)


@dataclass
class DavisActorModelCfg(RslRlModelCfg):
  """Actor cfg with DAVIS-specific annealing / aux weights (paper Table X + App. D-B)."""

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True
  # 论文未给出 k0/k1 / EMA 阈值 → 可配置合理默认。
  anneal_k0: int = 1000
  anneal_k1: int = 8000
  anneal_ema_threshold: float = 0.5
  aux_geo_weight_ball: float = 1.0
  aux_geo_weight_goal: float = 1.0
  aux_vis_weight: float = 0.2
  deployment_beta: float | None = None
  distribution_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "class_name": "rsl_rl.modules:GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
      "std_range": (0.05, 2.0),
    }
  )
  class_name: str = "src.tasks.davis.rl.davis_actor:DavisActorModel"


@dataclass
class DavisAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO + AMP hyperparams: DAVIS Table X + football_jax-scale throughput knobs.

  Depth CNN dominates wall-clock. With num_envs=4096 and horizon=24 the old
  (epochs=5, mini=4) used batch 24576 → ~465 ms/update on 4090; 20 updates ≈
  learn time. epochs=2 / mini=8 keeps batch ≈12288 (peak imgs/s in bench) and
  cuts CNN update volume ~3.5× while preserving on-policy sample size.
  """

  num_learning_epochs: int = 2
  num_mini_batches: int = 8
  learning_rate: float = 1.0e-3
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.01
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  share_cnn_encoders: bool = False
  # Paper Table X AMP coef 0.8; task/style lerp keeps task dominant early.
  # Mix soccer kick clips + walk_straight so chase gait is not kick-only.
  amp_motion_files: tuple[str, ...] = _AMP_FILES
  amp_motion_weights: tuple[float, ...] = _AMP_WEIGHTS
  amp_motion_labels: tuple[str, ...] = _AMP_LABELS
  amp_joint_names: tuple[str, ...] = WBOT_JOINT_NAMES  # 20-DoF; neck masked
  amp_key_body_names: tuple[str, ...] = WBOT_KEY_BODY_NAMES
  amp_reward_coefficient: float = 0.8
  # More task (chase) vs style — stand+look was farming AMP style.
  amp_task_reward_lerp: float = 0.92
  amp_discriminator_hidden_dims: tuple[int, ...] = (1024, 512, 256)
  amp_discriminator_type: str = "mlp"
  amp_part_gate_dim: int = 128
  amp_replay_buffer_size: int = 100_000
  amp_preload_transitions: int = 500_000
  amp_transition_dt: float = 0.02
  # football_jax configs/wbot20dof.yaml grad_pen_coef: 5.0
  amp_gradient_penalty: float = 5.0
  amp_motion_velocity_threshold: float = 0.8
  amp_standing_command_threshold: float = 0.1
  amp_root_dir: str = str(_REPO_ROOT)
  # 22 actions = 20 body + 2 neck.
  minimum_action_std: tuple[float, ...] = (0.05,) * 22
  class_name: str = "src.tasks.davis.rl.davis_amp_ppo:DavisAmpPPO"


@dataclass
class DavisShootRunnerCfg(RslRlOnPolicyRunnerCfg):
  seed: int = 42
  # football_jax runner.horizon_length / DAVIS Table X
  num_steps_per_env: int = 24
  max_iterations: int = 20_000  # paper
  save_interval: int = 100
  experiment_name: str = "davis_shoot_wbot_20dof"
  run_name: str = ""
  logger: Literal["tensorboard", "wandb"] = "tensorboard"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  upload_model: bool = False
  class_name: str = "OnPolicyRunner"
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      # aux_gt is consumed inside DavisActorModel for annealing only.
      # amp is read by DavisAmpPPO.act / process_env_step (not fed to actor MLP).
      "actor": ("actor_proprio", "actor_depth", "aux_gt"),
      "critic": ("critic",),
    }
  )
  actor: DavisActorModelCfg = field(default_factory=DavisActorModelCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="rsl_rl.models:MLPModel",
    )
  )
  algorithm: DavisAmpPpoAlgorithmCfg = field(default_factory=DavisAmpPpoAlgorithmCfg)


def wbot_20dof_davis_shoot_runner_cfg(*, sanity: bool = False) -> DavisShootRunnerCfg:
  cfg = DavisShootRunnerCfg()
  if sanity:
    cfg.max_iterations = 200
    cfg.save_interval = 50
    cfg.experiment_name = "davis_shoot_wbot_20dof_sanity"
    # Faster AMP preload for smoke tests.
    cfg.algorithm.amp_preload_transitions = 20_000
  return cfg
