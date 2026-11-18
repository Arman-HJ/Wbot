from mjlab.tasks.registry import register_mjlab_task

from src.tasks.ase.config.adam_sp.env_cfgs import (
  adam_sp_ase_flat_env_cfg,
  adam_sp_ase_rough_env_cfg,
  adam_sp_ase_transfer_flat_env_cfg,
  adam_sp_ase_transfer_rough_env_cfg,
)
from src.tasks.ase.config.adam_sp.rl_cfg import (
  adam_sp_ase_runner_cfg,
  adam_sp_ase_transfer_runner_cfg,
)
from src.tasks.ase.rl.ase_runner import RslRlAseOnPolicyRunner
from src.tasks.ase.rl.ase_transfer_runner import RslRlAseTransferOnPolicyRunner


def _register_adam_sp_ase_tasks() -> None:
  # Phase-1: skill embedding pre-training
  register_mjlab_task(
    task_id="AdamSP-ASE-Rough",
    env_cfg=adam_sp_ase_rough_env_cfg(),
    play_env_cfg=adam_sp_ase_rough_env_cfg(play=True),
    rl_cfg=adam_sp_ase_runner_cfg(),
    runner_cls=RslRlAseOnPolicyRunner,
  )
  register_mjlab_task(
    task_id="AdamSP-ASE-Flat",
    env_cfg=adam_sp_ase_flat_env_cfg(),
    play_env_cfg=adam_sp_ase_flat_env_cfg(play=True),
    rl_cfg=adam_sp_ase_runner_cfg(),
    runner_cls=RslRlAseOnPolicyRunner,
  )
  # Phase-2: high-level task transfer (set low_level_checkpoint via --checkpoint
  # is NOT used; pass via agent override or edit rl_cfg / CLI later).
  register_mjlab_task(
    task_id="AdamSP-ASE-Transfer-Flat",
    env_cfg=adam_sp_ase_transfer_flat_env_cfg(),
    play_env_cfg=adam_sp_ase_transfer_flat_env_cfg(play=True),
    rl_cfg=adam_sp_ase_transfer_runner_cfg(),
    runner_cls=RslRlAseTransferOnPolicyRunner,
  )
  register_mjlab_task(
    task_id="AdamSP-ASE-Transfer-Rough",
    env_cfg=adam_sp_ase_transfer_rough_env_cfg(),
    play_env_cfg=adam_sp_ase_transfer_rough_env_cfg(play=True),
    rl_cfg=adam_sp_ase_transfer_runner_cfg(),
    runner_cls=RslRlAseTransferOnPolicyRunner,
  )


_register_adam_sp_ase_tasks()
