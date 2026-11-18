"""RL configuration for WBOT 20DoF suspended task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.suspended.config.wbot_12dof.rl_cfg import wbot_12dof_suspended_ppo_runner_cfg


def wbot_20dof_suspended_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = wbot_12dof_suspended_ppo_runner_cfg()
  cfg.experiment_name = "wbot_20dof_suspended"
  # Shared schedule loops every 96 s ≈ 200 iters; 3000 iters ≈ 15 loops.
  cfg.max_iterations = 3000
  return cfg
