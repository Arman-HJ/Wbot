"""Register DAVIS shooting tasks for wbot 20-DoF."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import wbot_20dof_davis_shoot_env_cfg
from .rl_cfg import wbot_20dof_davis_shoot_runner_cfg

register_mjlab_task(
  task_id="Davis-Shoot-Wbot-20DOF",
  env_cfg=wbot_20dof_davis_shoot_env_cfg(),
  play_env_cfg=wbot_20dof_davis_shoot_env_cfg(play=True),
  rl_cfg=wbot_20dof_davis_shoot_runner_cfg(),
)

register_mjlab_task(
  task_id="Davis-Shoot-Wbot-20DOF-Sanity",
  env_cfg=wbot_20dof_davis_shoot_env_cfg(sanity=True),
  play_env_cfg=wbot_20dof_davis_shoot_env_cfg(play=True, sanity=True),
  rl_cfg=wbot_20dof_davis_shoot_runner_cfg(sanity=True),
)
