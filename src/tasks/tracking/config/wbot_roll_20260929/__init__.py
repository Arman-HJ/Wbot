from mjlab.tasks.registry import register_mjlab_task
from .roll_cfg import roll_env_cfg, roll_rl_cfg
from .rl_cfg import GETUP_TRACKING_RUNNER_CLS
register_mjlab_task(task_id="Wbot-20Dof-Tracking-RollToProne", env_cfg=roll_env_cfg(), play_env_cfg=roll_env_cfg(play=True), rl_cfg=roll_rl_cfg(), runner_cls=GETUP_TRACKING_RUNNER_CLS)
