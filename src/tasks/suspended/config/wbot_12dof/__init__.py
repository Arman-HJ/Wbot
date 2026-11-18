from mjlab.tasks.registry import register_mjlab_task

from src.tasks.suspended.config.wbot_12dof.env_cfgs import wbot_12dof_suspended_env_cfg
from src.tasks.suspended.config.wbot_12dof.rl_cfg import wbot_12dof_suspended_ppo_runner_cfg
from src.tasks.suspended.rl import SuspendedOnPolicyRunner

register_mjlab_task(
  task_id="Wbot-12Dof-Suspended",
  env_cfg=wbot_12dof_suspended_env_cfg(),
  play_env_cfg=wbot_12dof_suspended_env_cfg(play=True),
  rl_cfg=wbot_12dof_suspended_ppo_runner_cfg(),
  runner_cls=SuspendedOnPolicyRunner,
)
