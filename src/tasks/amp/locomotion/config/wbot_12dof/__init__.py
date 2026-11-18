"""Register the WBOT 12-DOF walk-and-run AMP locomotion task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfg import wbot_12dof_walk_amp_env_cfg
from .rl_cfg import wbot_12dof_walk_amp_runner_cfg


register_mjlab_task(
  task_id="AMP-Walk-Run-Wbot-12DOF",
  env_cfg=wbot_12dof_walk_amp_env_cfg(),
  play_env_cfg=wbot_12dof_walk_amp_env_cfg(play=True),
  rl_cfg=wbot_12dof_walk_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
