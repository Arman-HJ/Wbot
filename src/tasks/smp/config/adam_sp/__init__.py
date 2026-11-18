from mjlab.tasks.registry import register_mjlab_task

from src.tasks.smp.config.adam_sp.env_cfgs import (
  adam_sp_smp_flat_env_cfg,
  adam_sp_smp_rough_env_cfg,
)
from src.tasks.smp.config.adam_sp.rl_cfg import adam_sp_smp_runner_cfg
from src.tasks.smp.rl.smp_runner import RslRlSmpOnPolicyRunner


def _register_adam_sp_smp_tasks() -> None:
  register_mjlab_task(
    task_id="AdamSP-SMP-Rough",
    env_cfg=adam_sp_smp_rough_env_cfg(),
    play_env_cfg=adam_sp_smp_rough_env_cfg(play=True),
    rl_cfg=adam_sp_smp_runner_cfg(),
    runner_cls=RslRlSmpOnPolicyRunner,
  )
  register_mjlab_task(
    task_id="AdamSP-SMP-Flat",
    env_cfg=adam_sp_smp_flat_env_cfg(),
    play_env_cfg=adam_sp_smp_flat_env_cfg(play=True),
    rl_cfg=adam_sp_smp_runner_cfg(),
    runner_cls=RslRlSmpOnPolicyRunner,
  )


_register_adam_sp_smp_tasks()
