"""RL configuration for Z1 velocity v3 legs task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from src.tasks.velocity_v3.config.adam_sp.rl_cfg import adam_sp_velocity_v3_ppo_runner_cfg


def z1_velocity_v3_legs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for Z1 legs+waist velocity v3 (same topology as adam_pro legs v3)."""
  cfg = adam_sp_velocity_v3_ppo_runner_cfg()
  cfg.experiment_name = "z1_velocity_v3_legs"
  cfg.wandb_project = "z1_velocity_v3_legs"
  return cfg
