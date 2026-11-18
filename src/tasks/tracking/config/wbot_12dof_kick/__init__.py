from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wbot_12dof_kick_env_cfg
from .rl_cfg import KICK_RUNNER_CLS, wbot_12dof_kick_ppo_runner_cfg

register_mjlab_task(
  task_id="Wbot-12Dof-Kick",
  env_cfg=wbot_12dof_kick_env_cfg(),
  play_env_cfg=wbot_12dof_kick_env_cfg(play=True),
  rl_cfg=wbot_12dof_kick_ppo_runner_cfg(),
  runner_cls=KICK_RUNNER_CLS,
)
