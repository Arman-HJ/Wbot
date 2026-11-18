from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  z1_velocity_v3_legs_flat_env_cfg,
  z1_velocity_v3_legs_rough_env_cfg,
)
from .rl_cfg import z1_velocity_v3_legs_ppo_runner_cfg

register_mjlab_task(
  task_id="Z1-Velocity-v3-Legs-Rough",
  env_cfg=z1_velocity_v3_legs_rough_env_cfg(),
  play_env_cfg=z1_velocity_v3_legs_rough_env_cfg(play=True),
  rl_cfg=z1_velocity_v3_legs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Z1-Velocity-v3-Legs-Flat",
  env_cfg=z1_velocity_v3_legs_flat_env_cfg(),
  play_env_cfg=z1_velocity_v3_legs_flat_env_cfg(play=True),
  rl_cfg=z1_velocity_v3_legs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
