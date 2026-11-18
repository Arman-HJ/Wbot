"""Batch-convert retargeted ``.pkl`` clips to mjlab ``.npz`` for AMP / tracking."""

from __future__ import annotations

from pathlib import Path

import tyro

from pkl_to_npz import main as pkl_to_npz_main

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent.parent
_DEFAULT_INPUT_DIR = (
  _WORKSPACE_ROOT
  / "Motion_retargeting"
  / "GMR"
  / "retargeted_datasets"
  / "agibot_x2"
  / "robot_stand_walk_run"
)
_DEFAULT_OUTPUT_DIR = (
  _REPO_ROOT / "src" / "assets" / "amp_motions" / "x2" / "robot_stand_walk_run"
)
_DEFAULT_STEMS = ("stand-walk", "walk-run", "run-walk", "walk-stand")


def batch_pkl_to_npz(
  input_dir: Path = _DEFAULT_INPUT_DIR,
  output_dir: Path = _DEFAULT_OUTPUT_DIR,
  stems: tuple[str, ...] = _DEFAULT_STEMS,
  robot: str = "agibot_x2_full",
  suffix: str = "_agibot_x2",
  skip_existing: bool = False,
) -> None:
  """Convert ``{stem}{suffix}.pkl`` in ``input_dir`` to ``.npz`` in ``output_dir``."""
  input_dir = input_dir.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()
  if not input_dir.is_dir():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

  output_dir.mkdir(parents=True, exist_ok=True)

  for stem in stems:
    name = f"{stem}{suffix}"
    input_file = input_dir / f"{name}.pkl"
    output_file = output_dir / f"{name}.npz"

    if not input_file.is_file():
      raise FileNotFoundError(f"Missing input pkl: {input_file}")
    if skip_existing and output_file.is_file():
      print(f"[SKIP] {output_file} already exists", flush=True)
      continue

    print(f"[INFO] {input_file} -> {output_file}", flush=True)
    pkl_to_npz_main(
      input_file=str(input_file),
      output_file=str(output_file),
      robot=robot,
    )


if __name__ == "__main__":
  tyro.cli(batch_pkl_to_npz)
