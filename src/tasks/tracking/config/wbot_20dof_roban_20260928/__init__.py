from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  wbot_20dof_getup_tracking_env_cfg,
)
from .rl_cfg import (
  GETUP_TRACKING_RUNNER_CLS,
  wbot_20dof_getup_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-Getup",
  env_cfg=wbot_20dof_getup_tracking_env_cfg(
    pd_gains="armature_pace", history_encoder=True
  ),
  play_env_cfg=wbot_20dof_getup_tracking_env_cfg(
    play=True, pd_gains="armature_pace", history_encoder=True
  ),
  rl_cfg=wbot_20dof_getup_tracking_ppo_runner_cfg(history_encoder=True),
  runner_cls=GETUP_TRACKING_RUNNER_CLS,
)
