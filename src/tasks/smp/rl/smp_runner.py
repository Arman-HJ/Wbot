"""mjlab SMP runner — ``rsl_rl.SMPRunner`` + AMP-v2 video helpers."""

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import SMPRunner

from src.tasks.amp_v2.rl.rsl_rl_amp_runner import RslRlAmpOnPolicyRunner


class RslRlSmpOnPolicyRunner(RslRlAmpOnPolicyRunner, SMPRunner):
  """mjlab SMP runner: SDS style reward logging + checkpoint / video helpers."""

  _is_mjlab_smp_runner = True
  _is_mjlab_amp_v2_runner = False
  _uses_rsl_rl_amp_env = True

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    SMPRunner.__init__(self, env, train_cfg, log_dir, device)
    self._uploaded_videos: set = set()
    self._patch_logger_video_upload()

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    symmetry = getattr(self.alg, "symmetry", None)
    if symmetry is not None:
      self.env.unwrapped._symmetry_actor_normalizer = self.alg.get_policy().obs_normalizer
    try:
      SMPRunner.learn(
        self,
        num_learning_iterations=num_learning_iterations,
        init_at_random_ep_len=init_at_random_ep_len,
      )
    finally:
      if symmetry is not None:
        self.env.unwrapped._symmetry_actor_normalizer = None


def is_smp_runner_cls(runner_cls: type | None) -> bool:
  if not isinstance(runner_cls, type):
    return False
  return bool(getattr(runner_cls, "_is_mjlab_smp_runner", False))
