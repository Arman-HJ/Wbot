"""RL configuration for Adam SP decoupled (dual-agent) velocity tracking.

Every agent-related parameter is exposed here so it can be tuned without
touching the base config classes. See ``mjlab/rl/config.py`` for the full
field documentation.
"""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from src.tasks.velocity_v3.rl.rl_cfg_decoupled import RslRlDecoupledRunnerCfg


def adam_sp_velocity_decoupled_ppo_runner_cfg() -> RslRlDecoupledRunnerCfg:
  """Create PPODecoupled runner configuration for the Adam SP decoupled task."""
  return RslRlDecoupledRunnerCfg(
    # ---- task / action dims ----
    num_actions_lower=12,
    num_actions_upper=17,
    obs_groups={
      "actor_lower": ("actor",),
      "actor_upper": ("actor",),
      "critic_lower": ("critic",),
      "critic_upper": ("critic",),
    },

    # ---- lower-body actor (legs / gait) ----
    actor_lower=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),

    # ---- upper-body actor (torso + arms / stability) ----
    actor_upper=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),

    # ---- critics ----
    critic_lower=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    critic_upper=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),

    # ---- PPO wrapper (class_name + shared defaults; per-agent blocks override) ----
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="PPODecoupled",
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
    ),

    # ---- lower-body PPO (overrides ``algorithm`` for the lower agent) ----
    algorithm_lower=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
    ),

    # ---- upper-body PPO (overrides ``algorithm`` for the upper agent) ----
    algorithm_upper=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
    ),

    # ---- runner / logging ----
    experiment_name="adam_sp_velocity_decoupled",
    seed=42,
    run_name="",
    logger="wandb",
    wandb_project="adam_sp_velocity_decoupled",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_001,
    resume=False,
    load_run=".*",
    load_checkpoint="model_.*.pt",
    clip_actions=None,
    upload_model=True,
  )
