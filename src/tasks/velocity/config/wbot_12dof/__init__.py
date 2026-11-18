from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  wbot_12dof_flat_env_cfg,
  wbot_12dof_rough_env_cfg,
)
from .rl_cfg import wbot_12dof_ppo_runner_cfg

register_mjlab_task(
  task_id="Wbot-12Dof-Rough",
  env_cfg=wbot_12dof_rough_env_cfg(),
  play_env_cfg=wbot_12dof_rough_env_cfg(play=True),
  rl_cfg=wbot_12dof_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Wbot-12Dof-Flat",
  env_cfg=wbot_12dof_flat_env_cfg(),
  play_env_cfg=wbot_12dof_flat_env_cfg(play=True),
  rl_cfg=wbot_12dof_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
