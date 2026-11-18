"""HIL runner (extends ``rsl_rl`` AMP v2 runner)."""

from __future__ import annotations

from src.tasks.amp_v2.rl.rsl_rl_amp_runner import RslRlAmpOnPolicyRunner


class HilAmpOnPolicyRunner(RslRlAmpOnPolicyRunner):
  """HIL hybrid training: motion tracking + AMP with shared actor obs."""

  _is_mjlab_hil_runner = True


def is_hil_runner_cls(runner_cls) -> bool:
  if isinstance(runner_cls, type):
    return bool(getattr(runner_cls, "_is_mjlab_hil_runner", False))
  return False
