from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  wbot_20dof_backflip_tracking_env_cfg,
  wbot_20dof_fallgetup1_tracking_env_cfg,
  wbot_20dof_mj_dance_tracking_env_cfg,
  wbot_20dof_victory1_tracking_env_cfg,
  wbot_20dof_victory2_tracking_env_cfg,
)
from .rl_cfg import E1_MIMIC_RUNNER_CLS, wbot_20dof_e1_mimic_ppo_runner_cfg

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-Victory2",
  env_cfg=wbot_20dof_victory2_tracking_env_cfg(pd_gains="armature_soft_ankle_pace"),
  play_env_cfg=wbot_20dof_victory2_tracking_env_cfg(
    play=True, pd_gains="armature_soft_ankle_pace"
  ),
  rl_cfg=wbot_20dof_e1_mimic_ppo_runner_cfg(
    experiment_name="wbot_20dof_victory2", history_encoder=True
  ),
  runner_cls=E1_MIMIC_RUNNER_CLS,
)

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-FallGetUp1",
  env_cfg=wbot_20dof_fallgetup1_tracking_env_cfg(pd_gains="armature_soft_ankle_pace"),
  play_env_cfg=wbot_20dof_fallgetup1_tracking_env_cfg(
    play=True, pd_gains="armature_soft_ankle_pace"
  ),
  rl_cfg=wbot_20dof_e1_mimic_ppo_runner_cfg(
    experiment_name="wbot_20dof_fallgetup1", history_encoder=True
  ),
  runner_cls=E1_MIMIC_RUNNER_CLS,
)

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-Victory1",
  env_cfg=wbot_20dof_victory1_tracking_env_cfg(),
  play_env_cfg=wbot_20dof_victory1_tracking_env_cfg(play=True),
  rl_cfg=wbot_20dof_e1_mimic_ppo_runner_cfg(experiment_name="wbot_20dof_victory1"),
  runner_cls=E1_MIMIC_RUNNER_CLS,
)

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-MJDance",
  env_cfg=wbot_20dof_mj_dance_tracking_env_cfg(),
  play_env_cfg=wbot_20dof_mj_dance_tracking_env_cfg(play=True),
  rl_cfg=wbot_20dof_e1_mimic_ppo_runner_cfg(experiment_name="wbot_20dof_mj_dance"),
  runner_cls=E1_MIMIC_RUNNER_CLS,
)

register_mjlab_task(
  task_id="Wbot-20Dof-Tracking-Backflip",
  env_cfg=wbot_20dof_backflip_tracking_env_cfg(),
  play_env_cfg=wbot_20dof_backflip_tracking_env_cfg(play=True),
  rl_cfg=wbot_20dof_e1_mimic_ppo_runner_cfg(experiment_name="wbot_20dof_backflip"),
  runner_cls=E1_MIMIC_RUNNER_CLS,
)
