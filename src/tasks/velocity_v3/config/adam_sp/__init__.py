from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  adam_sp_velocity_v3_flat_env_cfg,
  adam_sp_velocity_v3_rough_env_cfg,
)
from .rl_cfg import adam_sp_velocity_v3_ppo_runner_cfg

register_mjlab_task(
  task_id="AdamSP-Velocity-v3-Rough",
  env_cfg=adam_sp_velocity_v3_rough_env_cfg(),
  play_env_cfg=adam_sp_velocity_v3_rough_env_cfg(play=True),
  rl_cfg=adam_sp_velocity_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="AdamSP-Velocity-v3-Flat",
  env_cfg=adam_sp_velocity_v3_flat_env_cfg(),
  play_env_cfg=adam_sp_velocity_v3_flat_env_cfg(play=True),
  rl_cfg=adam_sp_velocity_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
