"""Resolve play ``--run`` values to paths under logs/rsl_rl/<experiment>/."""

from __future__ import annotations

from pathlib import Path

from mjlab.tasks.registry import load_rl_cfg

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"


def resolve_run(task_id: str, run: str | None) -> str | None:
  """Accept a run name (optionally ``run/model_X.pt``); an existing path passes through."""
  if run is None or Path(run).expanduser().exists():
    return run
  candidate = LOG_ROOT / load_rl_cfg(task_id).experiment_name / run
  if not candidate.exists():
    raise FileNotFoundError(f"Neither {run!r} nor {candidate} exists")
  print(f"[INFO]: Run {run!r} → {candidate}")
  return str(candidate)
