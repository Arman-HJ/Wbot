from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import agibot_x2_velocity_v2_flat_env_cfg
from .rl_cfg import agibot_x2_velocity_v2_ppo_runner_cfg

register_mjlab_task(
  task_id="AgiBot-X2-Velocity-v2-Flat",
  env_cfg=agibot_x2_velocity_v2_flat_env_cfg(),
  play_env_cfg=agibot_x2_velocity_v2_flat_env_cfg(play=True),
  rl_cfg=agibot_x2_velocity_v2_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
