"""RL configuration for WBOT 20DoF e1_mimic tracking (kick-aligned PPO)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from src.tasks.tracking.rl.history_cfg import HistoryActorCfg, ReconPpoAlgorithmCfg


def wbot_20dof_e1_mimic_ppo_runner_cfg(
  *,
  experiment_name: str = "wbot_20dof_victory2",
  history_encoder: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """``history_encoder`` must match the env cfg flag of the same name."""
  actor_cls = HistoryActorCfg if history_encoder else RslRlModelCfg
  algorithm_cls = ReconPpoAlgorithmCfg if history_encoder else RslRlPpoAlgorithmCfg
  return RslRlOnPolicyRunnerCfg(
    actor=actor_cls(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=algorithm_cls(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
    logger="tensorboard",
    wandb_project=experiment_name,
  )


E1_MIMIC_RUNNER_CLS = MotionTrackingOnPolicyRunner
