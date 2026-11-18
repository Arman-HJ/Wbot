"""Register the WBOT getup AMP task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfg import wbot_20dof_getup_amp_env_cfg
from .rl_cfg import wbot_20dof_getup_amp_runner_cfg


register_mjlab_task(
  task_id="AMP-Getup-Wbot-20DOF",
  env_cfg=wbot_20dof_getup_amp_env_cfg(),
  play_env_cfg=wbot_20dof_getup_amp_env_cfg(play=True),
  rl_cfg=wbot_20dof_getup_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
