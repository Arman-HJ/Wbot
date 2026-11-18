#!/usr/bin/env python3
"""Batch-convert Adam SP locomotion clips to mjlab AMP ``.npz``.

Mirrors MimicKit ``dataset_humanoid_locomotion.yaml`` emphasis (run/walk weight 3.0)
using the Adam SP retargeted clips in ``dataset_adam_sp_locomotion.yaml``.

Sources (first match wins per clip):
  1. GMR ``.../pnd_adam_sp/robot_stand_walk_run/*.pkl``
  2. MimicKit ``data/motions/adam_sp/robot_stand_walk_run/*.pkl``

Outputs:
  ``src/assets/amp_motions/pnd_adam_sp/robot_stand_walk_run/{stem}_pnd_adam_sp.npz``
  ``dataset_adam_sp_locomotion.yaml`` (training weights for ``--amp-datasets``)

Usage (from unitree_rl_mjlab repo root)::

  python3 scripts/batch_adam_sp_locomotion.py
  python3 scripts/batch_adam_sp_locomotion.py --skip-existing
  python3 scripts/verify_amp_dataset.py --amp-dir src/assets/amp_motions/pnd_adam_sp/robot_stand_walk_run --robot pnd_adam_sp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.pkl_to_npz import batch_pkl_to_npz, convert_pkl_to_npz, check_pkl_joint_order  # noqa: E402

_MIMICKIT_ROOT = _REPO_ROOT.parent / "MimicKit"
_DEFAULT_GMR_DIR = (
  _REPO_ROOT.parent.parent
  / "Motion_retargeting"
  / "GMR"
  / "retargeted_datasets"
  / "pnd_adam_sp"
  / "robot_stand_walk_run"
)
_DEFAULT_MIMICKIT_DIR = (
  _MIMICKIT_ROOT / "data" / "motions" / "adam_sp" / "robot_stand_walk_run"
)
_DEFAULT_OUTPUT_DIR = (
  _REPO_ROOT / "src" / "assets" / "amp_motions" / "pnd_adam_sp" / "robot_stand_walk_run"
)
DEFAULT_SUFFIX = "_pnd_adam_sp"

# Same stems/weights as MimicKit dataset_adam_sp_locomotion.yaml (humanoid run/walk = 3.0).
DEFAULT_CLIPS: tuple[tuple[str, float], ...] = (
  ("walk-run", 3.0),
  ("run-walk", 3.0),
  ("stand-walk", 3.0),
  ("walk-stand", 3.0),
)


def _resolve_dir(path: Path, label: str, *, required: bool) -> Path | None:
  p = path.expanduser()
  if not p.is_absolute():
    p = (_REPO_ROOT / p).resolve()
  else:
    p = p.resolve()
  if p.is_dir():
    return p
  if required:
    raise FileNotFoundError(f"{label} is not a directory: {p}")
  return None


def _find_source_pkl(search_dirs: list[Path], stem: str, suffix: str) -> Path | None:
  candidates = [
    f"{stem}{suffix}.pkl",
    f"{stem} {suffix}.pkl",
    f"{stem}_{suffix}.pkl",
  ]
  for base in search_dirs:
    if base is None or not base.is_dir():
      continue
    for name in candidates:
      path = base / name
      if path.is_file():
        return path
    for path in sorted(base.glob(f"{stem}*{suffix}*.pkl")):
      return path
  return None


def _write_dataset_yaml(out_yaml: Path, entries: list[tuple[str, float]]) -> None:
  lines = [
    "motions:",
    "# Adam SP locomotion for unitree_rl_mjlab AMP v2 / HIL.",
    "# Weights mirror MimicKit dataset_humanoid_locomotion.yaml (run/walk = 3.0).",
    "# Train: --amp-datasets \"walk-run_pnd_adam_sp:3.0,run-walk_pnd_adam_sp:3.0,...\"",
  ]
  for stem, weight in entries:
    lines.append(f'  - file: "{stem}{DEFAULT_SUFFIX}.npz"')
    lines.append(f"    weight: {weight}")
  out_yaml.parent.mkdir(parents=True, exist_ok=True)
  out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"Wrote dataset yaml: {out_yaml}")


def batch_convert(
  *,
  gmr_dir: Path | None,
  mimickit_dir: Path | None,
  output_dir: Path,
  clips: tuple[tuple[str, float], ...],
  suffix: str,
  skip_existing: bool,
  check_joint_order: bool,
  output_fps: int,
  device: str,
  check_only: bool,
) -> None:
  search_dirs = [d for d in (gmr_dir, mimickit_dir) if d is not None]
  if not search_dirs:
    raise FileNotFoundError("No input directories (GMR or MimicKit).")

  output_dir.mkdir(parents=True, exist_ok=True)
  yaml_entries: list[tuple[str, float]] = []
  staging_dir = output_dir / ".pkl_staging"
  staging_dir.mkdir(parents=True, exist_ok=True)

  for stem, weight in clips:
    src = _find_source_pkl(search_dirs, stem, suffix)
    if src is None:
      raise FileNotFoundError(
        f"No source pkl for '{stem}{suffix}' under {search_dirs}. "
        "Run GMR retarget (Motion_retargeting/GMR/cmd.sh) or MimicKit "
        "tools/amp_npy_to_mimickit/batch_adam_sp_locomotion.py first."
      )

    out_name = f"{stem}{suffix}"
    out_npz = output_dir / f"{out_name}.npz"
    staged_pkl = staging_dir / f"{out_name}.pkl"

    if check_joint_order:
      check_pkl_joint_order(
        src,
        robot="pnd_adam_sp",
        gmr_xml=None,
        repo_root=_REPO_ROOT,
      )

    if check_only:
      print(f"[check ok] {src} -> {out_npz.name}")
      continue

    if skip_existing and out_npz.is_file():
      print(f"[SKIP] {out_npz} exists", flush=True)
      yaml_entries.append((stem, weight))
      continue

    if not staged_pkl.exists() or staged_pkl.resolve() != src.resolve():
      staged_pkl.write_bytes(src.read_bytes())

    print(f"[INFO] {src} -> {out_npz}", flush=True)
    convert_pkl_to_npz(
      input_file=str(staged_pkl),
      output_file=str(out_npz),
      robot="pnd_adam_sp",
      output_fps=output_fps,
      device=device,
      check_joint_order=False,
    )
    yaml_entries.append((stem, weight))

  if check_only:
    print("Check-only: no npz written.")
    return

  _write_dataset_yaml(output_dir / "dataset_adam_sp_locomotion.yaml", yaml_entries)
  amp_datasets = ",".join(f"{stem}{suffix}:{weight}" for stem, weight in yaml_entries)
  print(f"\nTrain example:\n  --amp-datasets \"{amp_datasets}\"")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert Adam SP GMR/MimicKit locomotion pkls to mjlab AMP npz."
  )
  parser.add_argument("--gmr-dir", type=Path, default=_DEFAULT_GMR_DIR)
  parser.add_argument("--mimickit-dir", type=Path, default=_DEFAULT_MIMICKIT_DIR)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
  parser.add_argument("--output-fps", type=int, default=50)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--skip-existing", action="store_true")
  parser.add_argument("--no-check-joint-order", action="store_true")
  parser.add_argument("--check-only", action="store_true")
  args = parser.parse_args()

  gmr_dir = _resolve_dir(args.gmr_dir, "--gmr-dir", required=False)
  mimickit_dir = _resolve_dir(args.mimickit_dir, "--mimickit-dir", required=False)

  batch_convert(
    gmr_dir=gmr_dir,
    mimickit_dir=mimickit_dir,
    output_dir=args.output_dir.expanduser().resolve(),
    clips=DEFAULT_CLIPS,
    suffix=args.suffix,
    skip_existing=args.skip_existing,
    check_joint_order=not args.no_check_joint_order,
    output_fps=args.output_fps,
    device=args.device,
    check_only=args.check_only,
  )


if __name__ == "__main__":
  main()
