"""Register the WBOT walk-and-run AMP locomotion task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfg import wbot_20dof_walk_amp_env_cfg
from .rl_cfg import wbot_20dof_walk_amp_runner_cfg


register_mjlab_task(
  task_id="AMP-Walk-Run-Wbot-20DOF",
  env_cfg=wbot_20dof_walk_amp_env_cfg(pd_gains="armature_pace"),
  play_env_cfg=wbot_20dof_walk_amp_env_cfg(play=True, pd_gains="armature_pace"),
  rl_cfg=wbot_20dof_walk_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="AMP-Walk-Run-Wbot-20DOF-ArmaturePD",
  env_cfg=wbot_20dof_walk_amp_env_cfg(pd_gains="armature"),
  play_env_cfg=wbot_20dof_walk_amp_env_cfg(play=True, pd_gains="armature"),
  rl_cfg=wbot_20dof_walk_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
