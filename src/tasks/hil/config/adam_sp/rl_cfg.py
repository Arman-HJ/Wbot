"""PND Adam SP HIL runner config."""

from __future__ import annotations

from src.tasks.amp_v2.config.adam_sp.rl_cfg import DEFAULT_AMP_DATA_DIR
from src.tasks.amp_v2.rl.rl_cfg import (
  AmpV2AlgorithmCfg,
  AmpV2RunnerCfg,
  DiscriminatorCfg,
  MlpModelCfg,
  OptimizerCfg,
  PpoTrainCfg,
  build_rsl_rl_amp_train_cfg,
)

__all__ = [
  "DEFAULT_AMP_DATA_DIR",
  "AmpV2RunnerCfg",
  "adam_sp_hil_runner_cfg",
  "build_rsl_rl_amp_train_cfg",
]


def adam_sp_hil_runner_cfg() -> AmpV2RunnerCfg:
  """HIL hybrid: motion tracking + AMP velocity task (50/50 task/disc mix)."""
  return AmpV2RunnerCfg(
    experiment_name="adam_sp_hil",
    wandb_project="adam_sp_hil",
    default_amp_data_dir=str(DEFAULT_AMP_DATA_DIR),
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=30_001,
    actor=MlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      action_std=0.05,
    ),
    critic=MlpModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    discriminator=DiscriminatorCfg(
      hidden_dims=(512, 256, 128),
      reward_scale=2.0,
      loss_type="BCEWithLogits",
    ),
    optimizer=OptimizerCfg(type="sgd", learning_rate=1e-4),
    algorithm=AmpV2AlgorithmCfg(
      ppo=PpoTrainCfg(
        clip_param=0.2,
        entropy_coef=0.0,
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
      task_reward_weight=0.5,
      disc_reward_weight=0.5,
      disc_trunk_weight_decay=1e-4,
      disc_logits_weight_decay=1e-2,
      disc_grad_penalty=2.0,
      symmetry_cfg={
        "use_data_augmentation": False,
        "data_augmentation_func": (
          "src.tasks.velocity_v3.rl.symmetry:compute_symmetric_states"
        ),
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.2,
      },
    ),
  )
