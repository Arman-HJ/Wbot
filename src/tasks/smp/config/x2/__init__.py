from mjlab.tasks.registry import register_mjlab_task

from src.tasks.smp.config.x2.env_cfgs import (
  agibot_x2_smp_flat_env_cfg,
  agibot_x2_smp_rough_env_cfg,
)
from src.tasks.smp.config.x2.rl_cfg import agibot_x2_smp_runner_cfg
from src.tasks.smp.rl.smp_runner import RslRlSmpOnPolicyRunner


def _register_x2_smp_tasks() -> None:
  register_mjlab_task(
    task_id="AgiBot-X2-SMP-Rough",
    env_cfg=agibot_x2_smp_rough_env_cfg(),
    play_env_cfg=agibot_x2_smp_rough_env_cfg(play=True),
    rl_cfg=agibot_x2_smp_runner_cfg(),
    runner_cls=RslRlSmpOnPolicyRunner,
  )
  register_mjlab_task(
    task_id="AgiBot-X2-SMP-Flat",
    env_cfg=agibot_x2_smp_flat_env_cfg(),
    play_env_cfg=agibot_x2_smp_flat_env_cfg(play=True),
    rl_cfg=agibot_x2_smp_runner_cfg(),
    runner_cls=RslRlSmpOnPolicyRunner,
  )


_register_x2_smp_tasks()
