"""ASE runner configs (phase-1 pretrain / phase-2 transfer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.tasks.amp_v2.rl.rl_cfg import (
  AdaptiveStyleRewardScaleCfg,
  AmpV2AlgorithmCfg,
  AmpV2RunnerCfg,
  DiscriminatorCfg,
  MlpModelCfg,
  OptimizerCfg,
  PpoTrainCfg,
  _build_mlp_model_cfg,
  _rsl_rl_loss_type,
  ppo_symmetry_cfg,
)
from src.tasks.amp_v2.mdp.motion_loader import resolve_amp_datasets


@dataclass
class AseParametersCfg:
  """ASE-specific hyperparameters (paper / MimicKit defaults)."""

  latent_dim: int = 64
  latent_steps_min: int = 1
  latent_steps_max: int = 150
  mi_reward_weight: float = 0.5  # enc_reward_weight (β)
  diversity_weight: float = 0.01  # w_div
  diversity_tar: float = 1.0  # MimicKit diversity target ratio
  encoder_scale: float = 1.0  # κ in vMF / MI reward
  enc_weight_decay: float = 0.0
  enc_learning_rate: float = 5e-5
  enc_hidden_dims: tuple[int, ...] = (512, 256)
  # False: MimicKit ``clamp(μ·z, 0)`` (paper eq.10 log-q term).
  # True: map cosine to [0,1] via 0.5(cos+1).
  mi_hypersphere_reward_shift: bool = False


@dataclass
class AseAlgorithmCfg(AmpV2AlgorithmCfg):
  """AMP algorithm knobs + ASE skill discovery."""

  ase: AseParametersCfg = field(default_factory=AseParametersCfg)
  # MimicKit ASEAgent: task/disc/enc reward weights.
  task_reward_weight: float = 0.0
  disc_reward_weight: float = 0.5
  # Explicit L2 on disc logit weights (MimicKit disc_logit_reg).
  disc_logit_reg: float = 0.01


@dataclass
class AseRunnerCfg(AmpV2RunnerCfg):
  """Phase-1 ASE pre-training runner config."""

  experiment_name: str = "adam_sp_ase"
  wandb_project: str = "adam_sp_ase"
  algorithm: AseAlgorithmCfg = field(default_factory=AseAlgorithmCfg)  # type: ignore[assignment]


@dataclass
class AseTransferRunnerCfg:
  """Phase-2 high-level policy config (frozen low-level skill policy)."""

  seed: int = 42
  num_steps_per_env: int = 24
  max_iterations: int = 50_001
  save_interval: int = 100
  experiment_name: str = "adam_sp_ase_transfer"
  run_name: str = ""
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  wandb_project: str = "adam_sp_ase_transfer"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  upload_model: bool = True

  # Path to phase-1 checkpoint (low-level π + frozen discriminator).
  low_level_checkpoint: str = ""
  latent_dim: int = 64
  # High-level queries every ``latent_repeat`` env steps (paper: 5 at 30Hz LL / 6Hz HL).
  latent_repeat: int = 5
  freeze_discriminator: bool = True
  task_reward_weight: float = 0.5
  disc_reward_weight: float = 0.5

  actor: MlpModelCfg = field(
    default_factory=lambda: MlpModelCfg(hidden_dims=(512, 256, 128), action_std=1.0)
  )
  critic: MlpModelCfg = field(
    default_factory=lambda: MlpModelCfg(hidden_dims=(512, 256, 128))
  )
  optimizer: OptimizerCfg = field(
    default_factory=lambda: OptimizerCfg(type="adam", learning_rate=2e-5)
  )
  ppo: PpoTrainCfg = field(default_factory=PpoTrainCfg)


def build_rsl_rl_ase_train_cfg(
  runner_cfg: AseRunnerCfg | None = None,
  *,
  amp_data_path: str | Path | None = None,
  slow_down_factor: int = 1,
  datasets: dict[str, float] | None = None,
  inference_only: bool = False,
) -> dict:
  """Build flat train dict for ASE phase-1 (``RslRlAseOnPolicyRunner`` / ``PPOASE``)."""
  cfg = runner_cfg or AseRunnerCfg()
  if inference_only:
    data_dir = Path("")
    dataset_weights: dict[str, float] = {}
  else:
    if not amp_data_path and not cfg.default_amp_data_dir:
      raise ValueError(
        "AMP data path is required. Pass --amp-data-path or set default_amp_data_dir."
      )
    data_dir = Path(amp_data_path or cfg.default_amp_data_dir).expanduser().resolve()
    dataset_weights = resolve_amp_datasets(data_dir, datasets)

  alg = cfg.algorithm
  assert isinstance(alg, AseAlgorithmCfg)
  ppo = alg.ppo
  disc = cfg.discriminator
  ase = alg.ase
  ac_opt = cfg.optimizer
  disc_opt = cfg.disc_optimizer
  if disc_opt is None:
    disc_opt_type = "adamw" if ac_opt.type == "muon_adamw" else ac_opt.type
    disc_opt = OptimizerCfg(type=disc_opt_type, learning_rate=ac_opt.learning_rate)
  disc_epochs = (
    ppo.num_learning_epochs
    if alg.disc_num_learning_epochs is None
    else int(alg.disc_num_learning_epochs)
  )
  disc_mini_batches = (
    ppo.num_mini_batches
    if alg.disc_num_mini_batches is None
    else int(alg.disc_num_mini_batches)
  )
  adaptive_scale_cfg = asdict(disc.adaptive_reward_scale)
  task_w = float(alg.task_reward_weight)
  disc_w = float(alg.disc_reward_weight)
  # Keep task_style_lerp for AMPDiscriminator ctor compatibility; ASE rewards
  # use explicit MimicKit weights in PPOASE.process_env_step.
  total_w = task_w + disc_w
  task_style_lerp = task_w / total_w if total_w > 0.0 else 0.0

  return {
    "seed": cfg.seed,
    "device": "cpu",
    "multi_gpu": None,
    "inference_only": inference_only,
    "logger": cfg.logger,
    "wandb_project": cfg.wandb_project,
    "wandb_kwargs": {
      "project": cfg.wandb_project,
      "run_name_prefix": cfg.experiment_name,
      "use_sequential_run_name": False,
    },
    "clip_actions": cfg.clip_actions,
    "num_steps_per_env": cfg.num_steps_per_env,
    "save_interval": cfg.save_interval,
    "max_iterations": cfg.max_iterations,
    "upload_model": cfg.upload_model,
    "tensorboard_video_max_frames": 300,
    "dataset": {
      "amp_data_path": str(data_dir),
      "datasets": dataset_weights,
      "slow_down_factor": slow_down_factor,
    },
    "obs_groups": {
      "policy": ["actor", "skill"],
      "critic": ["critic", "skill"],
      "discriminator": ["disc"],
      "discriminator_demonstration": ["disc_demo"],
    },
    "actor": _build_mlp_model_cfg(cfg.actor, stochastic=True),
    "critic": _build_mlp_model_cfg(cfg.critic),
    "algorithm": {
      "class_name": "PPOASE",
      "gamma": ppo.gamma,
      "lam": ppo.lam,
      "clip_param": ppo.clip_param,
      "entropy_coef": ppo.entropy_coef,
      "value_loss_coef": ppo.value_loss_coef,
      "use_clipped_value_loss": ppo.use_clipped_value_loss,
      "max_grad_norm": ppo.max_grad_norm,
      "schedule": ppo.schedule,
      "desired_kl": ppo.desired_kl,
      "num_learning_epochs": ppo.num_learning_epochs,
      "num_mini_batches": ppo.num_mini_batches,
      "learning_rate": ac_opt.learning_rate,
      "optimizer": ac_opt.type,
      "muon_learning_rate": ac_opt.muon_learning_rate,
      "muon_momentum": ac_opt.muon_momentum,
      "muon_weight_decay": ac_opt.muon_weight_decay,
      "rnd_cfg": None,
      "symmetry_cfg": ppo_symmetry_cfg(alg.symmetry_cfg),
      "ase_cfg": {
        "latent_dim": ase.latent_dim,
        "skill_obs_group": "skill",
        "task_reward_weight": task_w,
        "disc_reward_weight": disc_w,
        "mi_reward_weight": ase.mi_reward_weight,
        "diversity_weight": ase.diversity_weight,
        "diversity_tar": ase.diversity_tar,
        "encoder_scale": ase.encoder_scale,
        "enc_weight_decay": ase.enc_weight_decay,
        "enc_learning_rate": ase.enc_learning_rate,
        "enc_hidden_dims": list(ase.enc_hidden_dims),
        "mi_hypersphere_reward_shift": ase.mi_hypersphere_reward_shift,
      },
      "amp_cfg": {
        "disc_obs_buffer_size": max(100, alg.num_disc_obs_steps * 10),
        "grad_penalty_scale": alg.disc_grad_penalty,
        "disc_trunk_weight_decay": alg.disc_trunk_weight_decay,
        "disc_linear_weight_decay": alg.disc_logits_weight_decay,
        "disc_logit_reg": float(alg.disc_logit_reg),
        "disc_learning_rate": disc_opt.learning_rate,
        "disc_optimizer": (
          "adamw" if disc_opt.type == "muon_adamw" else disc_opt.type
        ),
        "disc_optimizer_betas": disc_opt.betas,
        "disc_max_grad_norm": (
          ppo.max_grad_norm
          if alg.disc_max_grad_norm is None
          else alg.disc_max_grad_norm
        ),
        "disc_num_learning_epochs": disc_epochs,
        "disc_num_mini_batches": disc_mini_batches,
        "disc_update_interval": max(1, int(alg.disc_update_interval)),
        "disc_grad_penalty_interval": max(1, int(alg.disc_grad_penalty_interval)),
        "disc_grad_penalty_batch_frac": float(alg.disc_grad_penalty_batch_frac),
        "loss_type": _rsl_rl_loss_type(disc),
        "adaptive_style_reward_scale": adaptive_scale_cfg,
        "freeze_discriminator": False,
        "amp_discriminator": {
          "hidden_dims": list(disc.hidden_dims),
          "activation": cfg.actor.activation,
          "style_reward_scale": disc.reward_scale,
          "reward_score_scale": disc.reward_score_scale,
          "task_style_lerp": task_style_lerp,
        },
      },
    },
  }


def build_rsl_rl_ase_transfer_train_cfg(
  runner_cfg: AseTransferRunnerCfg | None = None,
) -> dict:
  """Build flat train dict for ASE phase-2 hierarchical transfer."""
  cfg = runner_cfg or AseTransferRunnerCfg()
  ppo = cfg.ppo
  opt = cfg.optimizer
  task_w = float(cfg.task_reward_weight)
  disc_w = float(cfg.disc_reward_weight)
  total_w = task_w + disc_w
  task_style_lerp = task_w / total_w if total_w > 0.0 else 0.5
  return {
    "seed": cfg.seed,
    "device": "cpu",
    "multi_gpu": None,
    "inference_only": False,
    "logger": cfg.logger,
    "wandb_project": cfg.wandb_project,
    "wandb_kwargs": {
      "project": cfg.wandb_project,
      "run_name_prefix": cfg.experiment_name,
      "use_sequential_run_name": False,
    },
    "clip_actions": cfg.clip_actions,
    "num_steps_per_env": cfg.num_steps_per_env,
    "save_interval": cfg.save_interval,
    "max_iterations": cfg.max_iterations,
    "upload_model": cfg.upload_model,
    "low_level_checkpoint": cfg.low_level_checkpoint,
    "latent_dim": cfg.latent_dim,
    "latent_repeat": cfg.latent_repeat,
    "freeze_discriminator": cfg.freeze_discriminator,
    "obs_groups": {
      "policy": ["actor"],
      "critic": ["critic"],
      "discriminator": ["disc"],
      "discriminator_demonstration": ["disc_demo"],
    },
    "actor": _build_mlp_model_cfg(cfg.actor, stochastic=True),
    "critic": _build_mlp_model_cfg(cfg.critic),
    "algorithm": {
      "class_name": "PPO",
      "gamma": ppo.gamma,
      "lam": ppo.lam,
      "clip_param": ppo.clip_param,
      "entropy_coef": ppo.entropy_coef,
      "value_loss_coef": ppo.value_loss_coef,
      "use_clipped_value_loss": ppo.use_clipped_value_loss,
      "max_grad_norm": ppo.max_grad_norm,
      "schedule": ppo.schedule,
      "desired_kl": ppo.desired_kl,
      "num_learning_epochs": ppo.num_learning_epochs,
      "num_mini_batches": ppo.num_mini_batches,
      "learning_rate": opt.learning_rate,
      "optimizer": opt.type,
      "rnd_cfg": None,
      "symmetry_cfg": None,
      "task_style_lerp": task_style_lerp,
    },
  }
