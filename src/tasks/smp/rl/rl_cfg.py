"""SMP runner configs (frozen TinyMDM prior + PPO)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.tasks.amp_v2.mdp.motion_loader import resolve_amp_datasets
from src.tasks.amp_v2.rl.rl_cfg import (
  AmpV2RunnerCfg,
  MlpModelCfg,
  OptimizerCfg,
  PpoTrainCfg,
  _build_mlp_model_cfg,
  ppo_symmetry_cfg,
)


_UNITREE_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_PRIOR_DIR = _UNITREE_ROOT / "logs" / "smp_prior" / "engineai_prior_env_100k"


@dataclass
class SmpAlgorithmCfg:
  """SMP-specific hyperparameters (MimicKit defaults)."""

  num_disc_obs_steps: int = 10
  task_reward_weight: float = 0.0
  smp_reward_weight: float = 1.0
  sds_loss_scale: float = 6.0
  smp_reward_scale: float = 1.0
  # Larger default: SDS runs once per rollout over the full buffer (MimicKit-aligned).
  smp_eval_batch_size: int = 16384
  diffusion_steps: tuple[int, ...] = (22, 15, 8)
  sds_normalizer_samples: float = 1e8
  symmetry_cfg: dict | None = None
  ppo: PpoTrainCfg = field(default_factory=PpoTrainCfg)


@dataclass
class SmpRunnerCfg(AmpV2RunnerCfg):
  """SMP policy training runner config."""

  experiment_name: str = "adam_sp_smp"
  wandb_project: str = "adam_sp_smp"
  algorithm: SmpAlgorithmCfg = field(default_factory=SmpAlgorithmCfg)  # type: ignore[assignment]
  # Paths to trained TinyMDM prior (filled by adam_sp rl_cfg / CLI).
  smp_prior_cfg: str = ""
  smp_prior_model: str = ""


def build_rsl_rl_smp_train_cfg(
  runner_cfg: SmpRunnerCfg | None = None,
  *,
  amp_data_path: str | Path | None = None,
  slow_down_factor: int = 1,
  datasets: dict[str, float] | None = None,
  inference_only: bool = False,
) -> dict:
  """Build flat train dict for ``RslRlSmpOnPolicyRunner`` / ``PPOSMP``."""
  cfg = runner_cfg or SmpRunnerCfg()
  if inference_only:
    data_dir = Path("")
    dataset_weights: dict[str, float] = {}
  else:
    if not amp_data_path and not cfg.default_amp_data_dir:
      raise ValueError(
        "AMP data path is required for state init. Pass --amp-data-path or set default_amp_data_dir."
      )
    data_dir = Path(amp_data_path or cfg.default_amp_data_dir).expanduser().resolve()
    dataset_weights = resolve_amp_datasets(data_dir, datasets)

  prior_cfg = cfg.smp_prior_cfg
  prior_model = cfg.smp_prior_model
  alg = cfg.algorithm
  assert isinstance(alg, SmpAlgorithmCfg)
  style_enabled = float(alg.smp_reward_weight) > 0.0
  if not inference_only and style_enabled:
    if not prior_cfg or not prior_model:
      raise ValueError(
        "SMP prior paths required when smp_reward_weight > 0. "
        "Set smp_prior_cfg / smp_prior_model or use smp_reward_weight=0 for task-only."
      )
    if not Path(prior_cfg).expanduser().is_file():
      raise FileNotFoundError(f"SMP prior config not found: {prior_cfg}")
    if not Path(prior_model).expanduser().is_file():
      raise FileNotFoundError(f"SMP prior model not found: {prior_model}")

  ppo = alg.ppo
  ac_opt = cfg.optimizer

  smp_cfg_block: dict | None = None
  if not inference_only:
    smp_cfg_block = {
      "task_reward_weight": float(alg.task_reward_weight),
      "smp_reward_weight": float(alg.smp_reward_weight),
      "sds_loss_scale": float(alg.sds_loss_scale),
      "smp_reward_scale": float(alg.smp_reward_scale),
      "smp_eval_batch_size": int(alg.smp_eval_batch_size),
      "diffusion_steps": list(alg.diffusion_steps),
      "sds_normalizer_samples": float(alg.sds_normalizer_samples),
      "prior_model_class": "src.tasks.smp.tinymdm.tinymdm_model:TinyMDMModel",
    }
    if style_enabled:
      smp_cfg_block["smp_prior_cfg"] = str(Path(prior_cfg).expanduser().resolve())
      smp_cfg_block["smp_prior_model"] = str(Path(prior_model).expanduser().resolve())

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
      "policy": ["actor"],
      "critic": ["critic"],
      "discriminator": ["disc"],
    },
    "actor": _build_mlp_model_cfg(cfg.actor, stochastic=True),
    "critic": _build_mlp_model_cfg(cfg.critic),
    "algorithm": {
      "class_name": "PPOSMP",
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
      "smp_cfg": smp_cfg_block,
    },
  }


__all__ = [
  "SmpAlgorithmCfg",
  "SmpRunnerCfg",
  "build_rsl_rl_smp_train_cfg",
  "_DEFAULT_PRIOR_DIR",
]
