#!/usr/bin/env python3
"""Verify AMP ``.npz`` files match training-time joint indexing (no Isaac sim).

Training loads expert joints via ``MjlabAMPLoader._load_motion_dict``:
for each env AMP joint name, take ``joint_positions[frame][dataset_joint_names.index(name)]``.

Run before ``train.py`` (after ``fix_npz_joint_order`` or re-export)::

  python scripts/verify_amp_dataset.py --amp-dir src/assets/amp_motions/x2/robot_stand_walk_run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.assets.robots import AGIBOT_X2_NO_HEAD_JOINT_EXPR  # noqa: E402
from src.tasks.amp_v2.mdp.motion_loader import (  # noqa: E402
  npz_to_amp_motion_dict,
  read_dataset_joint_names,
  resolve_motion_dataset_path,
)

_NO_HEAD_RE = re.compile(AGIBOT_X2_NO_HEAD_JOINT_EXPR)


def expected_amp_joint_names(dataset_joint_names: list[str]) -> list[str]:
  """Same filter as env ``no_head_joint_cfg`` + ``find_joints(..., preserve_order=True)``."""
  return [n for n in dataset_joint_names if _NO_HEAD_RE.match(n)]


def verify_npz(
  npz_path: Path,
  amp_joint_names: list[str],
) -> tuple[bool, str]:
  data = npz_to_amp_motion_dict(npz_path, expected_joint_names=amp_joint_names)
  dataset_joint_names = data["joints_list"]
  missing = [j for j in amp_joint_names if j not in dataset_joint_names]
  if missing:
    return False, f"missing in joint_names: {missing[:5]}{'...' if len(missing) > 5 else ''}"

  idx_map = [dataset_joint_names.index(j) if j in dataset_joint_names else None for j in amp_joint_names]
  if any(i is None for i in idx_map):
    return False, "idx_map has None entries"

  # Spot-check: right/left shoulder pitch should not be constant zero.
  jp = np.stack(data["joint_positions"], axis=0)
  mapped = jp[:, idx_map]
  r_idx = amp_joint_names.index("right_shoulder_pitch_joint")
  l_idx = amp_joint_names.index("left_shoulder_pitch_joint")
  r_std = float(mapped[:, r_idx].std())
  l_std = float(mapped[:, l_idx].std())
  if r_std < 1e-4 or l_std < 1e-4:
    return False, f"arm motion near-constant (std L={l_std:.2e} R={r_std:.2e}) — wrong column?"

  # Head columns in npz should exist but are not fed to AMP (29-dim expert).
  head_in_npz = [n for n in dataset_joint_names if n.startswith("head_")]
  return True, (
    f"ok: {mapped.shape[1]} amp joints, {len(head_in_npz)} head cols in npz (ignored), "
    f"shoulder_pitch std L={l_std:.3f} R={r_std:.3f}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Verify AMP npz vs training joint map")
  parser.add_argument(
    "--amp-dir",
    type=Path,
    default=_REPO_ROOT / "src/assets/amp_motions/x2/robot_stand_walk_run",
  )
  parser.add_argument(
    "--datasets",
    nargs="*",
    default=None,
    help="Dataset stems (no ext). Default: all .npz in amp-dir",
  )
  args = parser.parse_args()
  amp_dir = args.amp_dir.expanduser().resolve()
  if not amp_dir.is_dir():
    raise SystemExit(f"Not a directory: {amp_dir}")

  if args.datasets:
    names = list(args.datasets)
  else:
    names = sorted(p.stem for p in amp_dir.glob("*.npz") if not p.stem.endswith("_fixed"))

  # Build union joint list like MjlabAMPLoader when amp_joint_names not preset.
  joint_union: list[str] = []
  seen: set[str] = set()
  for name in names:
    path = resolve_motion_dataset_path(amp_dir, name)
    for j in read_dataset_joint_names(path, expected_joint_names=None):
      if j not in seen:
        seen.add(j)
        joint_union.append(j)

  amp_joint_names = expected_amp_joint_names(joint_union)
  print(f"AMP training joints (no head): {len(amp_joint_names)}")
  print(f"  first: {amp_joint_names[:3]} ... last: {amp_joint_names[-3:]}")

  all_ok = True
  for name in names:
    path = resolve_motion_dataset_path(amp_dir, name)
    ok, msg = verify_npz(path, amp_joint_names)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {path.name}: {msg}")
    all_ok = all_ok and ok

  print()
  if all_ok:
    print("Dataset OK for training loader. Next checks:")
    print("  1) train log: [INFO] AMP expert joints: 29 ...")
    print("  2) TensorBoard: Misc/Disc_Demo_Acc, Misc/Disc_Agent_Acc, Train/mean_style_reward")
    print("  3) scripts/play.py early checkpoint — gait plausible, arms not twisted")
    print("Old checkpoints trained on mis-ordered npz must be retrained.")
  else:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
