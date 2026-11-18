from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  agibot_x2_standing_legs_flat_env_cfg,
  agibot_x2_standing_legs_rough_env_cfg,
)
from .rl_cfg import agibot_x2_standing_legs_ppo_runner_cfg

register_mjlab_task(
  task_id="AgiBot-X2-Standing-Legs-Rough",
  env_cfg=agibot_x2_standing_legs_rough_env_cfg(),
  play_env_cfg=agibot_x2_standing_legs_rough_env_cfg(play=True),
  rl_cfg=agibot_x2_standing_legs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="AgiBot-X2-Standing-Legs-Flat",
  env_cfg=agibot_x2_standing_legs_flat_env_cfg(),
  play_env_cfg=agibot_x2_standing_legs_flat_env_cfg(play=True),
  rl_cfg=agibot_x2_standing_legs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
