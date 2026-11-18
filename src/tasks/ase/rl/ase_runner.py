"""ASE phase-1 runner (skill pre-training) — uses ``rsl_rl.ASERunner`` + ``LoggerASE``."""

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import ASERunner

from src.tasks.amp_v2.rl.rsl_rl_amp_runner import RslRlAmpOnPolicyRunner


class RslRlAseOnPolicyRunner(RslRlAmpOnPolicyRunner, ASERunner):
  """mjlab ASE runner: ``ASERunner`` logging + AMP-v2 checkpoint / video helpers."""

  _is_mjlab_ase_runner = True
  _is_mjlab_amp_v2_runner = False  # use ASE train-cfg path, not AMP-v2
  _uses_rsl_rl_amp_env = True

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # ASERunner sets LoggerASE; RslRlAmpOnPolicyRunner adds video upload patch.
    ASERunner.__init__(self, env, train_cfg, log_dir, device)
    self._uploaded_videos: set = set()
    self._patch_logger_video_upload()

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    symmetry = getattr(self.alg, "symmetry", None)
    if symmetry is not None:
      self.env.unwrapped._symmetry_actor_normalizer = self.alg.get_policy().obs_normalizer
    try:
      ASERunner.learn(
        self,
        num_learning_iterations=num_learning_iterations,
        init_at_random_ep_len=init_at_random_ep_len,
      )
    finally:
      if symmetry is not None:
        self.env.unwrapped._symmetry_actor_normalizer = None


def is_ase_runner_cls(runner_cls: type | None) -> bool:
  if not isinstance(runner_cls, type):
    return False
  return bool(getattr(runner_cls, "_is_mjlab_ase_runner", False))


def is_ase_transfer_runner_cls(runner_cls: type | None) -> bool:
  if not isinstance(runner_cls, type):
    return False
  return bool(getattr(runner_cls, "_is_mjlab_ase_transfer_runner", False))
