"""Tests for AMP v2 motion frame indexing."""

from __future__ import annotations

import torch

from src.tasks.amp_v2.mdp.motion_loader import (
  AmpMotionResetLibrary,
  resolve_dataset_name,
)


class _FakeLoader:
  dataset_names = ["stand-walk_pnd_adam_sp", "walk-run_pnd_adam_sp"]
  clip_starts = torch.tensor([0, 100], dtype=torch.long)
  clip_lengths = torch.tensor([100, 80], dtype=torch.long)
  all_reference_states = torch.zeros(180, 10)


def test_resolve_dataset_name() -> None:
  names = _FakeLoader.dataset_names
  assert resolve_dataset_name(names, "stand-walk_pnd_adam_sp") == "stand-walk_pnd_adam_sp"
  assert resolve_dataset_name(names, "stand-walk_pnd_adam_sp.npz") == "stand-walk_pnd_adam_sp"


def test_global_to_clip_local() -> None:
  library = AmpMotionResetLibrary(
    loader=_FakeLoader(), joint_dim=3, joint_names=[]
  )  # type: ignore[arg-type]
  global_idx = torch.tensor([0, 50, 99, 100, 150, 250])
  local = library.global_to_clip_local(
    global_idx, "stand-walk_pnd_adam_sp", fallback_length=100
  )
  assert local.tolist() == [0, 50, 99, 0, 50, 50]
