"""Wbot-20Dof-Tracking-VelStyle: velocity-conditioned whole-body style tracking."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wbot_20dof_vel_style_env_cfg
from .rl_cfg import VEL_STYLE_RUNNER_CLS, wbot_20dof_vel_style_ppo_runner_cfg

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-VelStyle",
  env_cfg=wbot_20dof_vel_style_env_cfg(),
  play_env_cfg=wbot_20dof_vel_style_env_cfg(play=True),
  rl_cfg=wbot_20dof_vel_style_ppo_runner_cfg(),
  runner_cls=VEL_STYLE_RUNNER_CLS,
)
