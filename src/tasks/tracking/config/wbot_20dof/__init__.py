from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import wbot_20dof_flat_tracking_env_cfg
from .rl_cfg import wbot_20dof_tracking_ppo_runner_cfg
from .grounded_env_cfg import grounded_start_tracking_env_cfg
from .support_env_cfg import support_tracking_env_cfg
from .support_runner import SupportTrackingRunner

_support_rl = wbot_20dof_tracking_ppo_runner_cfg()
_support_rl.experiment_name = "wbot_20dof_getup_tracking_support_v4"
_support_rl.actor.distribution_cfg["init_std"] = .3
_support_rl.algorithm.gamma = .995
_support_rl.algorithm.entropy_coef = .002
register_mjlab_task(
  task_id="Wbot-20DoF-Getup-Tracking-Support-V4",
  env_cfg=support_tracking_env_cfg(),
  play_env_cfg=support_tracking_env_cfg(play=True),
  rl_cfg=_support_rl,
  runner_cls=SupportTrackingRunner,
)

_grounded_rl = wbot_20dof_tracking_ppo_runner_cfg()
_grounded_rl.experiment_name = "wbot_20dof_getup_tracking_grounded"
register_mjlab_task(
  task_id="Wbot-20DoF-Getup-Tracking-Grounded-Start",
  env_cfg=grounded_start_tracking_env_cfg(),
  play_env_cfg=grounded_start_tracking_env_cfg(play=True),
  rl_cfg=_grounded_rl,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Wbot-20DoF-Getup-Tracking",
  env_cfg=wbot_20dof_flat_tracking_env_cfg(),
  play_env_cfg=wbot_20dof_flat_tracking_env_cfg(play=True),
  rl_cfg=wbot_20dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Wbot-20DoF-Getup-Tracking-No-State-Estimation",
  env_cfg=wbot_20dof_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=wbot_20dof_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=wbot_20dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
