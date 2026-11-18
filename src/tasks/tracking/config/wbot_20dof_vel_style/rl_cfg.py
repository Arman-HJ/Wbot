"""RL configuration for WBOT 20DoF velocity-conditioned style tracking.

Plain MLP actor/critic (paper Sec. III) — no history encoder / AMP disc.
"""

from mjlab.rl import (
  MjlabOnPolicyRunner,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def wbot_20dof_vel_style_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
    algorithm=RslRlPpoAlgorithmCfg(
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
    experiment_name="wbot_20dof_vel_style",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
    logger="tensorboard",
    wandb_project="wbot_20dof_vel_style",
  )


# Plain runner: export policy.onnx with velocity-command actor obs (no ref joints).
VEL_STYLE_RUNNER_CLS = MjlabOnPolicyRunner
