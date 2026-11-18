"""RL configuration for AgiBot X2 decoupled (dual-agent) velocity tracking."""

from src.tasks.velocity_v3.rl.rl_cfg_decoupled import RslRlDecoupledRunnerCfg


def agibot_x2_velocity_decoupled_ppo_runner_cfg() -> RslRlDecoupledRunnerCfg:
  """Create PPODecoupled runner configuration for the X2 decoupled task."""
  return RslRlDecoupledRunnerCfg(
    num_actions_lower=12,
    num_actions_upper=17,
    experiment_name="x2_velocity_decoupled",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100_001,
  )
