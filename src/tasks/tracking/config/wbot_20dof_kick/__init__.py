from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wbot_20dof_kick_env_cfg
from .rl_cfg import KICK_RUNNER_CLS, wbot_20dof_kick_ppo_runner_cfg

register_mjlab_task(
  task_id="Wbot-20Dof-Kick",
  env_cfg=wbot_20dof_kick_env_cfg(pd_gains="armature_soft_ankle_pace"),
  play_env_cfg=wbot_20dof_kick_env_cfg(play=True, pd_gains="armature_soft_ankle_pace"),
  rl_cfg=wbot_20dof_kick_ppo_runner_cfg(),
  runner_cls=KICK_RUNNER_CLS,
)
