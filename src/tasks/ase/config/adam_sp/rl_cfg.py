"""PND Adam SP ASE runner configs (phase-1 pretrain / phase-2 transfer)."""

from __future__ import annotations

from src.tasks.amp_v2.config.adam_sp.rl_cfg import DEFAULT_AMP_DATA_DIR
from src.tasks.amp_v2.rl.rl_cfg import (
  AdaptiveStyleRewardScaleCfg,
  DiscriminatorCfg,
  MlpModelCfg,
  OptimizerCfg,
  PpoTrainCfg,
)
from src.tasks.ase.rl.rl_cfg import (
  AseAlgorithmCfg,
  AseParametersCfg,
  AseRunnerCfg,
  AseTransferRunnerCfg,
)

# Tuned for engineai_locomotion (~3 clips), with MimicKit ASE algorithm choices.
_ASE_SMALL_LATENT_DIM = 32
_ASE_SMALL_LATENT_STEPS = (30, 150)


def adam_sp_ase_runner_cfg() -> AseRunnerCfg:
  """Phase-1: unsupervised skill embedding pre-training (ASE / MimicKit)."""
  return AseRunnerCfg(
    logger="tensorboard",
    experiment_name="adam_sp_ase",
    wandb_project="adam_sp_ase",
    default_amp_data_dir=str(DEFAULT_AMP_DATA_DIR),
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=100_001,
    actor=MlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      action_std=0.05,  # paper Σπ≈0.0025 variance → std≈0.05
    ),
    critic=MlpModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    discriminator=DiscriminatorCfg(
      hidden_dims=(512, 256),
      # MimicKit disc_reward_scale=2; style = scale * -log(1-D)
      reward_scale=2.0,
      # ASE paper eq.14 / MimicKit: BCE GAN + GP
      loss_type="BCEWithLogits",
      adaptive_reward_scale=AdaptiveStyleRewardScaleCfg(enabled=False),
    ),
    optimizer=OptimizerCfg(type="adam", learning_rate=2e-5),
    disc_optimizer=OptimizerCfg(
      type="adam",
      learning_rate=5e-5,
      betas=(0.0, 0.9),
    ),
    algorithm=AseAlgorithmCfg(
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
      # MimicKit: task/disc/enc = 0 / 0.5 / 0.5
      task_reward_weight=0.0,
      disc_reward_weight=0.5,
      disc_logit_reg=0.01,
      disc_trunk_weight_decay=0.0,
      # MimicKit disc optimizer weight_decay=1e-4 on logits
      disc_logits_weight_decay=1e-4,
      disc_grad_penalty=10.0,  # small-data: stronger than MimicKit's 5
      disc_max_grad_norm=0.0,
      disc_num_learning_epochs=2,  # MimicKit disc_epochs
      disc_update_interval=2,
      disc_grad_penalty_interval=1,
      disc_grad_penalty_batch_frac=1.0,
      ase=AseParametersCfg(
        latent_dim=_ASE_SMALL_LATENT_DIM,
        latent_steps_min=_ASE_SMALL_LATENT_STEPS[0],
        latent_steps_max=_ASE_SMALL_LATENT_STEPS[1],
        mi_reward_weight=0.5,  # MimicKit enc_reward_weight
        diversity_weight=0.01,  # MimicKit (small-data keep mild)
        diversity_tar=1.0,
        encoder_scale=1.0,
        enc_learning_rate=5e-5,
        enc_hidden_dims=(512, 256),
        mi_hypersphere_reward_shift=False,
      ),
    ),
  )


def adam_sp_ase_transfer_runner_cfg(
  low_level_checkpoint: str = "",
) -> AseTransferRunnerCfg:
  """Phase-2: high-level velocity task on frozen ASE low-level skills."""
  return AseTransferRunnerCfg(
    logger="tensorboard",
    experiment_name="adam_sp_ase_transfer",
    wandb_project="adam_sp_ase_transfer",
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=50_001,
    low_level_checkpoint=low_level_checkpoint,
    latent_dim=_ASE_SMALL_LATENT_DIM,
    latent_repeat=5,
    freeze_discriminator=True,
    task_reward_weight=0.5,
    disc_reward_weight=0.5,
    actor=MlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      action_std=1.0,  # explore near origin of unnormalized Z̄
    ),
    critic=MlpModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    optimizer=OptimizerCfg(type="adam", learning_rate=2e-5),
    ppo=PpoTrainCfg(
      clip_param=0.2,
      entropy_coef=0.01,
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
  )


__all__ = [
  "DEFAULT_AMP_DATA_DIR",
  "_ASE_SMALL_LATENT_DIM",
  "_ASE_SMALL_LATENT_STEPS",
  "adam_sp_ase_runner_cfg",
  "adam_sp_ase_transfer_runner_cfg",
]
