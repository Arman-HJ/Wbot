from mjlab.tasks.registry import register_mjlab_task

from src.tasks.hil.config.adam_sp.env_cfgs import adam_sp_hil_flat_env_cfg
from src.tasks.hil.config.adam_sp.rl_cfg import adam_sp_hil_runner_cfg
from src.tasks.hil.rl.hil_amp_runner import HilAmpOnPolicyRunner


def _register_adam_sp_hil_tasks() -> None:
  register_mjlab_task(
    task_id="AdamSP-HIL-Flat",
    env_cfg=adam_sp_hil_flat_env_cfg(),
    play_env_cfg=adam_sp_hil_flat_env_cfg(play=True),
    rl_cfg=adam_sp_hil_runner_cfg(),  # type: ignore[arg-type]
    runner_cls=HilAmpOnPolicyRunner,
  )


_register_adam_sp_hil_tasks()
