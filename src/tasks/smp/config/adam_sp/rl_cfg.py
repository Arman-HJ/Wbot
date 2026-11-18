"""PND Adam SP SMP runner configs."""

from __future__ import annotations

from pathlib import Path

from src.tasks.amp_v2.config.adam_sp.rl_cfg import DEFAULT_AMP_DATA_DIR
from src.tasks.amp_v2.rl.rl_cfg import MlpModelCfg, OptimizerCfg, PpoTrainCfg
from src.tasks.smp.rl.rl_cfg import SmpAlgorithmCfg, SmpRunnerCfg, _DEFAULT_PRIOR_DIR


def adam_sp_smp_runner_cfg(
  *,
  smp_prior_cfg: str | None = None,
  smp_prior_model: str | None = None,
) -> SmpRunnerCfg:
  """SMP policy: amp_v2 task rewards + frozen TinyMDM style prior (no disc)."""
  prior_dir = _DEFAULT_PRIOR_DIR
  return SmpRunnerCfg(
    logger="tensorboard",
    experiment_name="adam_sp_smp",
    wandb_project="adam_sp_smp",
    default_amp_data_dir=str(DEFAULT_AMP_DATA_DIR),
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=100_001,
    smp_prior_cfg=smp_prior_cfg or str(prior_dir / "diffusion_config.yaml"),
    smp_prior_model=smp_prior_model or str(prior_dir / "model_99999.pt"),
    # Match amp_v2 actor/critic / PPO defaults for fair task learning.
    actor=MlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      action_std=1.0,
    ),
    critic=MlpModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    optimizer=OptimizerCfg(type="adamw", learning_rate=1e-3),
    algorithm=SmpAlgorithmCfg(
      ppo=PpoTrainCfg(
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        schedule="adaptive",
        num_learning_epochs=5,
        num_mini_batches=4,
      ),
      num_disc_obs_steps=10,
      task_reward_weight=0.8,
      smp_reward_weight=0.2,
      sds_loss_scale=6.0,
      smp_reward_scale=1.0,
      smp_eval_batch_size=16384,
      diffusion_steps=(22, 15, 8),
    ),
  )


__all__ = [
  "DEFAULT_AMP_DATA_DIR",
  "adam_sp_smp_runner_cfg",
]
