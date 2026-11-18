from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  agibot_x2_velocity_v3_flat_enhanced_env_cfg,
  agibot_x2_velocity_v3_flat_env_cfg,
  agibot_x2_velocity_v3_rough_env_cfg,
)
from .rl_cfg import agibot_x2_velocity_v3_ppo_runner_cfg

register_mjlab_task(
  task_id="AgiBot-X2-Velocity-v3-Rough",
  env_cfg=agibot_x2_velocity_v3_rough_env_cfg(),
  play_env_cfg=agibot_x2_velocity_v3_rough_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="AgiBot-X2-Velocity-v3-Flat",
  env_cfg=agibot_x2_velocity_v3_flat_env_cfg(),
  play_env_cfg=agibot_x2_velocity_v3_flat_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="AgiBot-X2-Velocity-v3-Flat-Enhanced",
  env_cfg=agibot_x2_velocity_v3_flat_enhanced_env_cfg(),
  play_env_cfg=agibot_x2_velocity_v3_flat_enhanced_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
