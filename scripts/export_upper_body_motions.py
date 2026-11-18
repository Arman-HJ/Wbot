#!/usr/bin/env python3
"""Export FALCON-style upper-body motion packs for Adam SP.

Primary source is the FALCON X2 ``motions_upper`` pack (same 14 joint names /
YPR wrist order as Adam SP). Joint axes already match, so mapping is
name-based copy + clip to Adam SP limits.

Writes::

  <out>/
    playlist.yaml
    <slug>_upper14.npy   # (T, 14) absolute rad
    <slug>_meta.yaml

Usage::

  python scripts/export_upper_body_motions.py --from-falcon
  python scripts/export_upper_body_motions.py \\
    --npz src/assets/amp_motions/pnd_adam_sp/feet_gait/forward_walk_woman_ss_pnd_adam_sp.npz
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "src/assets/upper_motions/pnd_adam_sp"
DEFAULT_FALCON_DIR = (
  ROOT.parent / "FALCON/humanoidverse/data/deploy/x2/motions_upper"
)

UPPER_JOINT_NAMES = [
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
]

# Adam SP XML joint ranges (left/right shoulder_roll are mirrored).
ADAM_SP_UPPER_LIMITS: dict[str, tuple[float, float]] = {
  "left_shoulder_pitch_joint": (-3.613, 2.042),
  "left_shoulder_roll_joint": (-0.628, 2.793),
  "left_shoulder_yaw_joint": (-2.583, 2.583),
  "left_elbow_joint": (-2.496, 0.209),
  "left_wrist_yaw_joint": (-2.67, 2.67),
  "left_wrist_pitch_joint": (-0.96, 0.96),
  "left_wrist_roll_joint": (-0.96, 0.96),
  "right_shoulder_pitch_joint": (-3.613, 2.042),
  "right_shoulder_roll_joint": (-2.793, 0.628),
  "right_shoulder_yaw_joint": (-2.583, 2.583),
  "right_elbow_joint": (-2.496, 0.209),
  "right_wrist_yaw_joint": (-2.67, 2.67),
  "right_wrist_pitch_joint": (-0.96, 0.96),
  "right_wrist_roll_joint": (-0.96, 0.96),
}

# X2 FALCON upper14 already uses the same hinge sense as Adam SP (no flips).
ADAM_SP_UPPER_SIGN: dict[str, float] = {name: 1.0 for name in UPPER_JOINT_NAMES}


def _slug(path: Path) -> str:
  name = path.stem
  name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
  return name[:48]


def _clip_to_adam_sp(upper: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
  clipped = upper.astype(np.float32, copy=True)
  clip_counts: dict[str, int] = {}
  signs = np.array([ADAM_SP_UPPER_SIGN[n] for n in UPPER_JOINT_NAMES], dtype=np.float32)
  clipped *= signs
  for i, name in enumerate(UPPER_JOINT_NAMES):
    lo, hi = ADAM_SP_UPPER_LIMITS[name]
    before = clipped[:, i].copy()
    clipped[:, i] = np.clip(before, lo, hi)
    n_clip = int(np.count_nonzero(before != clipped[:, i]))
    if n_clip:
      clip_counts[name] = n_clip
  return clipped, clip_counts


def _write_clip(
  out: Path,
  slug: str,
  upper: np.ndarray,
  *,
  fps: float,
  extra_meta: dict | None = None,
) -> dict:
  out.mkdir(parents=True, exist_ok=True)
  mapped, clip_counts = _clip_to_adam_sp(upper)
  np.save(out / f"{slug}_upper14.npy", mapped)
  meta = {
    "slug": slug,
    "fps": float(fps),
    "num_frames": int(mapped.shape[0]),
    "duration_s": float(mapped.shape[0] / max(fps, 1e-6)),
    "units": "rad absolute (not residual)",
    "upper14_joint_names": UPPER_JOINT_NAMES,
    "target_robot": "pnd_adam_sp",
    "sign_flips": {k: v for k, v in ADAM_SP_UPPER_SIGN.items() if v < 0},
    "clipped_frames": clip_counts,
  }
  if extra_meta:
    meta.update(extra_meta)
  with open(out / f"{slug}_meta.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)
  return {
    "slug": slug,
    "upper14": f"{slug}_upper14.npy",
    "meta": f"{slug}_meta.yaml",
    "fps": float(fps),
    "num_frames": int(mapped.shape[0]),
  }


def export_npz(npz_path: Path, out: Path, slug: str | None = None) -> dict:
  with np.load(npz_path, allow_pickle=True) as data:
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    if "joint_names" in data:
      joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
    else:
      raise KeyError(f"{npz_path}: missing joint_names")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0

  name_to_col = {n: i for i, n in enumerate(joint_names)}
  missing = [n for n in UPPER_JOINT_NAMES if n not in name_to_col]
  if missing:
    raise KeyError(f"{npz_path}: missing joints {missing}")
  cols = [name_to_col[n] for n in UPPER_JOINT_NAMES]
  upper = joint_pos[:, cols].astype(np.float32)
  slug = slug or _slug(npz_path)
  entry = _write_clip(
    out,
    slug,
    upper,
    fps=fps,
    extra_meta={"source_npz": str(npz_path)},
  )
  entry["source_npz"] = npz_path.name
  return entry


def export_falcon_pack(falcon_dir: Path, out: Path) -> list[dict]:
  playlist_path = falcon_dir / "playlist.yaml"
  if not playlist_path.is_file():
    raise FileNotFoundError(f"FALCON playlist not found: {playlist_path}")
  with open(playlist_path, encoding="utf-8") as handle:
    src = yaml.safe_load(handle) or {}
  clips = src.get("clips") or []
  if not clips:
    raise ValueError(f"Empty FALCON playlist: {playlist_path}")

  playlist = []
  for clip in clips:
    slug = str(clip.get("slug") or Path(clip["upper14"]).stem)
    upper_path = falcon_dir / str(clip["upper14"])
    if not upper_path.is_file():
      raise FileNotFoundError(upper_path)
    upper = np.load(upper_path).astype(np.float32)
    if upper.ndim != 2 or upper.shape[1] != len(UPPER_JOINT_NAMES):
      raise ValueError(f"{upper_path}: expected (T, 14), got {upper.shape}")
    fps = float(clip.get("fps", 30.0))
    entry = _write_clip(
      out,
      slug,
      upper,
      fps=fps,
      extra_meta={
        "source_pack": str(falcon_dir),
        "source_key": clip.get("source_key"),
        "source_upper14": clip.get("upper14"),
        "mapping": (
          "FALCON X2 upper14 -> Adam SP upper14 by joint name. "
          "Same YPR wrist order; hinge axes match; no sign flips."
        ),
      },
    )
    if clip.get("source_key"):
      entry["source_key"] = clip["source_key"]
    playlist.append(entry)
  return playlist


def _write_playlist(out: Path, playlist: list[dict], robot: str) -> None:
  with open(out / "playlist.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(
      {
        "robot": robot,
        "framework": "falcon",
        "source": "FALCON X2 motions_upper (ACCAD), remapped to Adam SP",
        "policy_dt": 0.02,
        "upper14_joint_names": UPPER_JOINT_NAMES,
        "clips": playlist,
      },
      handle,
      sort_keys=False,
      allow_unicode=True,
    )
  (out / "README.md").write_text(
    """# Adam SP upper-body motion pack (FALCON tracking)

Absolute joint angles in **radians**, 14 arm DoF (no waist).

Mapped from FALCON X2 `motions_upper` (ACCAD lift / put-down / gesture clips).
Joint names and hinge sense already match Adam SP; values are clipped to
Adam SP XML limits.

| file | shape | use |
|------|-------|-----|
| `*_upper14.npy` | `(T, 14)` | standing `ref_upper_dof_pos` |
| `playlist.yaml` | — | clip list for `UpperBodyMotionCommand` |
""",
    encoding="utf-8",
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--from-falcon",
    nargs="?",
    const=str(DEFAULT_FALCON_DIR),
    default=None,
    help="FALCON motions_upper directory (default if flag given with no path)",
  )
  parser.add_argument(
    "--npz",
    type=Path,
    nargs="+",
    default=None,
    help="Optional extra mjlab motion npz files",
  )
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--robot", type=str, default="pnd_adam_sp")
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Remove existing files in --out before writing",
  )
  args = parser.parse_args()

  if args.from_falcon is None and not args.npz:
    args.from_falcon = str(DEFAULT_FALCON_DIR)

  out = args.out.expanduser().resolve()
  if args.clean and out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True, exist_ok=True)

  playlist: list[dict] = []
  if args.from_falcon:
    falcon_dir = Path(args.from_falcon).expanduser().resolve()
    falcon_clips = export_falcon_pack(falcon_dir, out)
    playlist.extend(falcon_clips)
    for entry in falcon_clips:
      print(f"[falcon] {entry['slug']}  T={entry['num_frames']} fps={entry['fps']}")

  if args.npz:
    for npz_path in args.npz:
      npz_path = npz_path.expanduser().resolve()
      entry = export_npz(npz_path, out)
      playlist.append(entry)
      print(f"[npz] {entry['slug']}  T={entry['num_frames']} fps={entry['fps']}")

  _write_playlist(out, playlist, args.robot)
  print(f"wrote {out / 'playlist.yaml'} ({len(playlist)} clips)")


if __name__ == "__main__":
  main()
