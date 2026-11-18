#!/usr/bin/env python3
"""Convert legged_lab G1 AMP ``.pkl`` clips to Adam SP mjlab ``.npz``.

legged_lab stores G1 lab-format motion (wxyz root quat, G1 lab DOF order).
This script remaps joint order to PND Adam SP GMR layout, writes a staging
GMR pkl (xyzw), then calls ``pkl_to_npz`` for full-body AMP expert clips.

Source (default): ``rl/legged_lab/.../MotionData/g1_29dof/amp/walk_and_run/*.pkl``
Output (default): ``src/assets/amp_motions/pnd_adam_sp/walk_and_run/{stem}_pnd_adam_sp.npz``

Usage (repo root)::

  python3 scripts/batch_adam_sp_from_legged_lab.py
  python3 scripts/batch_adam_sp_from_legged_lab.py --skip-existing
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.pkl_to_npz import convert_pkl_to_npz  # noqa: E402

_DEFAULT_LEGGED_LAB_DIR = (
  _REPO_ROOT.parent
  / "legged_lab"
  / "source"
  / "legged_lab"
  / "legged_lab"
  / "data"
  / "MotionData"
  / "g1_29dof"
  / "amp"
  / "walk_and_run"
)
_DEFAULT_OUTPUT_DIR = (
  _REPO_ROOT / "src" / "assets" / "amp_motions" / "pnd_adam_sp" / "walk_and_run"
)

ADAM_SP_AMP_SUFFIX = "_pnd_adam_sp"

# Mirrors ``legged_lab/.../g1_amp_env_cfg.py`` motion_data_weights (1.0 each).
LEGGED_LAB_AMP_CLIPS: tuple[tuple[str, float], ...] = (
  ("B10_-__Walk_turn_left_45_stageii", 1.0),
  ("B11_-__Walk_turn_left_135_stageii", 1.0),
  ("B13_-__Walk_turn_right_90_stageii", 1.0),
  ("B14_-__Walk_turn_right_45_t2_stageii", 1.0),
  ("B15_-__Walk_turn_around_stageii", 1.0),
  ("B22_-__side_step_left_stageii", 1.0),
  ("B23_-__side_step_right_stageii", 1.0),
  ("B4_-_Stand_to_Walk_backwards_stageii", 1.0),
  ("B9_-__Walk_turn_left_90_stageii", 1.0),
  ("C11_-_run_turn_left_90_stageii", 1.0),
  ("C12_-_run_turn_left_45_stageii", 1.0),
  ("C13_-_run_turn_left_135_stageii", 1.0),
  ("C14_-_run_turn_right_90_stageii", 1.0),
  ("C15_-_run_turn_right_45_stageii", 1.0),
  ("C16_-_run_turn_right_135_stageii", 1.0),
  ("C17_-_run_change_direction_stageii", 1.0),
  ("C1_-_stand_to_run_stageii", 1.0),
  ("C3_-_run_stageii", 1.0),
  ("C4_-_run_to_walk_a_stageii", 1.0),
  ("C5_-_walk_to_run_stageii", 1.0),
  ("C6_-_stand_to_run_backwards_stageii", 1.0),
  ("C8_-_run_backwards_to_stand_stageii", 1.0),
  ("C9_-_run_backwards_turn_run_forward_stageii", 1.0),
  ("Walk_B10_-_Walk_turn_left_45_stageii", 1.0),
  ("Walk_B13_-_Walk_turn_right_45_stageii", 1.0),
  ("Walk_B15_-_Walk_turn_around_stageii", 1.0),
  ("Walk_B16_-_Walk_turn_change_stageii", 1.0),
  ("Walk_B22_-_Side_step_left_stageii", 1.0),
  ("Walk_B23_-_Side_step_right_stageii", 1.0),
  ("Walk_B4_-_Stand_to_Walk_Back_stageii", 1.0),
)

DEFAULT_SUFFIX = ADAM_SP_AMP_SUFFIX

# Isaac Lab G1 lab DOF order (``g1_29dof.yaml`` ``lab_dof_names``).
G1_LAB_DOF_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

# PND Adam SP MuJoCo qpos order (GMR ``joint_order.json``).
ADAM_SP_DOF_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)

_G1_TO_ADAM_SP_INDEX = tuple(
  G1_LAB_DOF_NAMES.index(name) for name in ADAM_SP_DOF_NAMES
)


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
  q = np.asarray(quat_wxyz, dtype=np.float32)
  return q[:, [1, 2, 3, 0]]


def _remap_dof(dof_g1_lab: np.ndarray) -> np.ndarray:
  dof = np.asarray(dof_g1_lab, dtype=np.float64)
  return dof[:, _G1_TO_ADAM_SP_INDEX]


def g1_lab_pkl_to_gmr_pkl(g1_lab_path: Path) -> dict:
  """Convert legged_lab G1 lab pkl to GMR-style pkl for ``pkl_to_npz``."""
  with g1_lab_path.open("rb") as f:
    data = pickle.load(f)
  required = ("fps", "root_pos", "root_rot", "dof_pos")
  for key in required:
    if key not in data:
      raise KeyError(f"{g1_lab_path} missing key {key!r}")

  return {
    "fps": float(data["fps"]),
    "root_pos": np.asarray(data["root_pos"], dtype=np.float64),
    "root_rot": _wxyz_to_xyzw(data["root_rot"]),
    "dof_pos": _remap_dof(data["dof_pos"]),
    "joint_names": list(ADAM_SP_DOF_NAMES),
    "dof_column_order": "mujoco_qpos",
    "robot": "pnd_adam_sp",
  }


def _foot_body_indices(body_names: list[str]) -> list[int]:
  idx = [i for i, n in enumerate(body_names) if "ankle_roll_link" in n]
  return idx or list(range(len(body_names)))


def measure_npz_foot_min_z(npz_path: Path) -> float:
  """Lowest ankle-roll body height in an expert ``.npz`` (meters)."""
  with np.load(npz_path, allow_pickle=True) as z:
    body_names = [str(x) for x in z["body_names"].tolist()]
    foot_idx = _foot_body_indices(body_names)
    bp = np.asarray(z["body_pos_w"], dtype=np.float64)
  return float(bp[:, foot_idx, 2].min())


def align_gmr_root_height(gmr: dict, dz: float) -> dict:
  """Lift root trajectory so FK feet clear the ground (Adam SP vs G1 skeleton)."""
  if abs(dz) < 1e-9:
    return gmr
  out = dict(gmr)
  root_pos = np.asarray(out["root_pos"], dtype=np.float64).copy()
  root_pos[:, 2] += float(dz)
  out["root_pos"] = root_pos
  return out


def _convert_gmr_pkl_to_npz(
  *,
  staged_pkl: Path,
  gmr: dict,
  out_npz: Path,
  output_fps: int,
  device: str,
) -> float:
  with staged_pkl.open("wb") as f:
    pickle.dump(gmr, f)
  convert_pkl_to_npz(
    input_file=str(staged_pkl),
    output_file=str(out_npz),
    robot="pnd_adam_sp",
    output_fps=output_fps,
    device=device,
    check_joint_order=False,
  )
  return measure_npz_foot_min_z(out_npz)


def align_npz_clip_to_ground(
  npz_path: Path,
  *,
  ground_margin: float = 0.03,
  staged_pkl: Path | None = None,
  gmr: dict | None = None,
  output_fps: int = 50,
  device: str = "cuda:0",
) -> float:
  """Re-lift root and re-export ``.npz`` so feet are above ``ground_margin``.

  G1 DOF remapped onto Adam SP often penetrates the ground (~12 cm worst case).
  Shifting ``root_pos.z`` and re-running FK keeps joints / body poses consistent.
  """
  min_z = measure_npz_foot_min_z(npz_path)
  if min_z >= ground_margin:
    return 0.0
  dz = ground_margin - min_z
  if gmr is None or staged_pkl is None:
    raise ValueError("align_npz_clip_to_ground needs gmr + staged_pkl to reconvert.")
  aligned = align_gmr_root_height(gmr, dz)
  new_min = _convert_gmr_pkl_to_npz(
    staged_pkl=staged_pkl,
    gmr=aligned,
    out_npz=npz_path,
    output_fps=output_fps,
    device=device,
  )
  if new_min < ground_margin - 1e-3:
    print(
      f"[WARN] {npz_path.name}: foot min z={new_min:.4f} after +{dz:.4f} m root lift",
      flush=True,
    )
  return dz


def _write_dataset_yaml(out_yaml: Path, entries: list[tuple[str, float]]) -> None:
  lines = [
    "motions:",
    "# Adam SP AMP clips adapted from legged_lab G1 walk_and_run (30 clips).",
    "# Train: --amp-data-path=src/assets/amp_motions/pnd_adam_sp/walk_and_run",
  ]
  for stem, weight in entries:
    lines.append(f'  - file: "{stem}{DEFAULT_SUFFIX}.npz"')
    lines.append(f"    weight: {weight}")
  out_yaml.parent.mkdir(parents=True, exist_ok=True)
  out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"Wrote dataset yaml: {out_yaml}")


def batch_convert(
  *,
  input_dir: Path,
  output_dir: Path,
  clips: tuple[tuple[str, float], ...],
  suffix: str,
  skip_existing: bool,
  align_only: bool,
  ground_margin: float,
  output_fps: int,
  device: str,
) -> None:
  input_dir = input_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  staging_dir = output_dir / ".g1_staging"
  staging_dir.mkdir(parents=True, exist_ok=True)

  yaml_entries: list[tuple[str, float]] = []
  missing: list[str] = []

  for stem, weight in clips:
    src = input_dir / f"{stem}.pkl"
    out_npz = output_dir / f"{stem}{suffix}.npz"
    staged_pkl = staging_dir / f"{stem}{suffix}.pkl"

    if align_only:
      if not out_npz.is_file():
        print(f"[WARN] missing npz for align: {out_npz.name}", flush=True)
        continue
      if not src.is_file():
        missing.append(stem)
        continue
      gmr = g1_lab_pkl_to_gmr_pkl(src)
      dz = align_npz_clip_to_ground(
        out_npz,
        ground_margin=ground_margin,
        staged_pkl=staged_pkl,
        gmr=gmr,
        output_fps=output_fps,
        device=device,
      )
      print(f"[ALIGN] {out_npz.name}: dz={dz:.4f} m", flush=True)
      yaml_entries.append((stem, weight))
      continue

    if not src.is_file():
      missing.append(stem)
      continue

    if skip_existing and out_npz.is_file():
      print(f"[SKIP] {out_npz.name}", flush=True)
      yaml_entries.append((stem, weight))
      continue

    gmr = g1_lab_pkl_to_gmr_pkl(src)

    print(f"[INFO] {src.name} -> {out_npz.name}", flush=True)
    _convert_gmr_pkl_to_npz(
      staged_pkl=staged_pkl,
      gmr=gmr,
      out_npz=out_npz,
      output_fps=output_fps,
      device=device,
    )
    dz = align_npz_clip_to_ground(
      out_npz,
      ground_margin=ground_margin,
      staged_pkl=staged_pkl,
      gmr=gmr,
      output_fps=output_fps,
      device=device,
    )
    if dz > 0:
      print(f"[INFO] ground align {out_npz.name}: +{dz:.4f} m root z", flush=True)
    yaml_entries.append((stem, weight))

  if missing:
    print(
      f"[WARN] {len(missing)} clips missing under {input_dir} (skipped).",
      flush=True,
    )

  if not yaml_entries:
    raise FileNotFoundError(f"No clips converted from {input_dir}")

  _write_dataset_yaml(output_dir / "dataset_adam_sp_walk_and_run.yaml", yaml_entries)
  amp_datasets = ",".join(f"{stem}{suffix}:{weight}" for stem, weight in yaml_entries)
  print(f"\nConverted {len(yaml_entries)} clips.")
  print(f"Train example:\n  --amp-data-path={output_dir}\n  --amp-datasets \"{amp_datasets}\"")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert legged_lab G1 AMP pkls to Adam SP mjlab npz."
  )
  parser.add_argument("--input-dir", type=Path, default=_DEFAULT_LEGGED_LAB_DIR)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
  parser.add_argument("--output-fps", type=int, default=50)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--skip-existing", action="store_true")
  parser.add_argument(
    "--align-only",
    action="store_true",
    help="Only re-run ground alignment on existing npz (no full reconvert).",
  )
  parser.add_argument(
    "--ground-margin",
    type=float,
    default=0.03,
    help="Minimum foot height above z=0 after alignment (meters).",
  )
  args = parser.parse_args()

  batch_convert(
    input_dir=args.input_dir.expanduser(),
    output_dir=args.output_dir.expanduser().resolve(),
    clips=LEGGED_LAB_AMP_CLIPS,
    suffix=args.suffix,
    skip_existing=args.skip_existing,
    align_only=args.align_only,
    ground_margin=args.ground_margin,
    output_fps=args.output_fps,
    device=args.device,
  )


if __name__ == "__main__":
  main()
