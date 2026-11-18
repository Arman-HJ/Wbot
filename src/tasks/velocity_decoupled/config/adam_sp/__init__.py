from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  adam_sp_velocity_decoupled_flat_env_cfg,
  adam_sp_velocity_decoupled_rough_env_cfg,
)
from .rl_cfg import adam_sp_velocity_decoupled_ppo_runner_cfg
from src.tasks.velocity_decoupled.rl.runner import VelocityDecoupledOnPolicyRunner

register_mjlab_task(
  task_id="AdamSP-Velocity-Decoupled-Flat",
  env_cfg=adam_sp_velocity_decoupled_flat_env_cfg(),
  play_env_cfg=adam_sp_velocity_decoupled_flat_env_cfg(play=True),
  rl_cfg=adam_sp_velocity_decoupled_ppo_runner_cfg(),
  runner_cls=VelocityDecoupledOnPolicyRunner,
)

register_mjlab_task(
  task_id="AdamSP-Velocity-Decoupled-Rough",
  env_cfg=adam_sp_velocity_decoupled_rough_env_cfg(),
  play_env_cfg=adam_sp_velocity_decoupled_rough_env_cfg(play=True),
  rl_cfg=adam_sp_velocity_decoupled_ppo_runner_cfg(),
  runner_cls=VelocityDecoupledOnPolicyRunner,
)
