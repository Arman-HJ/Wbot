#!/usr/bin/env python3
"""Build paper-protocol VelStyle motion set for WBOT 20DoF.

Paper (arXiv:2611.20888) motion composition
-------------------------------------------
1. Record human mocap (authors' private capture; not released).
2. Retarget with GMR → robot full-body reference per frame:
   joint_pos/vel + body_{pos,quat,lin_vel,ang_vel}_w; pelvis = anchor.
3. Five semantic clips:
   - random_walk  : main training clip (covers a range of planar velocities)
   - backward     : held-out / in-dist eval
   - side         : held-out / in-dist eval
   - slow         : held-out / in-dist eval
   - jog          : in-dist eval (not used as held-out in Table IV)

Protocol:
  - Generalization: train ONLY on random_walk; evaluate with velocity profiles
    from held-out backward / side / slow.
  - Ablation B1: also train on left-right mirrored random_walk.
  - Ablation S: smooth base pos + frame-to-frame base rotation with Savitzky-Golay
    (window=100), keep joint trajectories; rebuild body kinematics.

This script cannot recreate Agile Robots' private mocap. It builds the same
*protocol roles* from existing WBOT AMP locomotion clips (already GMR-like /
retargeted), then expands them to full-body tracking npz via
``motion_npz_to_tracking.expand``.

WBOT source mapping (documented approximations where noted)::

  random_walk ← walk_run_amp/walk.npz
      (54s, vx in [-0.75,0.90], ~25% vx<0 — closest to paper random walk)
  side        ← walk_run_amp/side_l.npz
  jog         ← walk_run_amp/run.npz
  backward    ← time-reverse of walk_straight_a (approx. backward gait)
  slow        ← 2× time-stretch of walk_straight_b (approx. slow walk)

Usage::

  python scripts/prepare_vel_style_paper_motions.py
  python scripts/prepare_vel_style_paper_motions.py --also-smoothed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.signal import savgol_filter

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.motion_npz_to_tracking import (  # noqa: E402
  DEFAULT_XML,
  GMR_WBOT_JOINTS,
  expand,
  report_quality,
  save,
)

AMP_DIR = _REPO / "src/assets/amp_motions/wbot_20dof/walk_run_amp"
OUT_DIR = _REPO / "src/assets/motions/wbot_20dof/vel_style"

# Paper role → (source amp npz | synthetic recipe, notes)
PAPER_ROLES = {
  "random_walk": {
    "source": "walk.npz",
    "kind": "direct",
    "notes": "Paper main train clip. WBOT walk.npz covers multi-direction planar vel.",
  },
  "side": {
    "source": "side_l.npz",
    "kind": "direct",
    "notes": "Paper held-out / in-dist side walking.",
  },
  "jog": {
    "source": "run.npz",
    "kind": "direct",
    "notes": "Paper in-dist jog (Table III/V). Not a Table IV held-out.",
  },
  "backward": {
    "source": "walk_straight_a.npz",
    "kind": "time_reverse",
    "notes": (
      "APPROXIMATION: paper recorded a dedicated backward clip. "
      "We time-reverse a straight walk and rebuild velocities (common biped hack)."
    ),
  },
  "slow": {
    "source": "walk_straight_b.npz",
    "kind": "time_stretch",
    "stretch": 2.0,
    "notes": (
      "APPROXIMATION: paper recorded a dedicated slow walk. "
      "We 2× time-stretch a straight walk and rebuild velocities."
    ),
  },
}


def _fps_of(data: dict[str, np.ndarray], default: float = 50.0) -> float:
  if "fps" not in data:
    return float(default)
  v = np.asarray(data["fps"]).reshape(-1)
  return float(v[0]) if v.size else float(default)


def _load_amp(path: Path) -> dict[str, np.ndarray]:
  data = np.load(path, allow_pickle=True)
  out = {k: data[k] for k in data.files}
  if "joint_names" not in out:
    out["joint_names"] = np.asarray(GMR_WBOT_JOINTS)
  return out


def _root_from_amp(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """Return (root_pos (T,3), root_quat_wxyz (T,4))."""
  if "body_pos_w" in data and data["body_pos_w"].ndim == 3:
    return data["body_pos_w"][:, 0].astype(np.float64), data["body_quat_w"][:, 0].astype(
      np.float64
    )
  raise KeyError(f"expected body_pos_w in amp clip, keys={list(data)}")


def _fix_quat_wxyz(q: np.ndarray) -> np.ndarray:
  q = q / np.linalg.norm(q, axis=1, keepdims=True)
  for t in range(1, len(q)):
    if np.dot(q[t - 1], q[t]) < 0.0:
      q[t] = -q[t]
  return q


def _to_tracking(
  joint_pos: np.ndarray,
  joint_names: list[str],
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  fps: float,
) -> dict[str, np.ndarray]:
  motion = expand(
    joint_pos.astype(np.float64),
    joint_names,
    root_pos.astype(np.float64),
    _fix_quat_wxyz(root_quat.astype(np.float64)),
    float(fps),
    xml=DEFAULT_XML,
  )
  report_quality(
    motion["joint_pos"],
    list(motion["joint_names"]),
    motion["_joint_range"],
    float(fps),
    body_ang_vel_w=motion["body_ang_vel_w"],
  )
  return motion


def convert_direct(src: Path, fps: float | None = None) -> dict[str, np.ndarray]:
  data = _load_amp(src)
  joint_names = [str(n) for n in data["joint_names"]]
  root_pos, root_quat = _root_from_amp(data)
  use_fps = float(fps if fps is not None else _fps_of(data))
  return _to_tracking(data["joint_pos"], joint_names, root_pos, root_quat, use_fps)


def convert_time_reverse(src: Path) -> dict[str, np.ndarray]:
  """Reverse frame order; rebuild all velocities via expand()."""
  data = _load_amp(src)
  joint_names = [str(n) for n in data["joint_names"]]
  root_pos, root_quat = _root_from_amp(data)
  fps = _fps_of(data)
  return _to_tracking(
    data["joint_pos"][::-1].copy(),
    joint_names,
    root_pos[::-1].copy(),
    root_quat[::-1].copy(),
    fps,
  )


def convert_time_stretch(src: Path, stretch: float) -> dict[str, np.ndarray]:
  """Spatially keep poses; stretch duration by ``stretch`` via pose interpolation."""
  if stretch <= 1.0:
    raise ValueError("stretch must be > 1")
  data = _load_amp(src)
  joint_names = [str(n) for n in data["joint_names"]]
  root_pos, root_quat = _root_from_amp(data)
  fps = _fps_of(data)
  t_old = np.arange(len(root_pos), dtype=np.float64)
  n_new = int(round(len(root_pos) * stretch))
  t_new = np.linspace(0.0, len(root_pos) - 1, n_new)
  joint_pos = np.stack(
    [np.interp(t_new, t_old, data["joint_pos"][:, j]) for j in range(data["joint_pos"].shape[1])],
    axis=1,
  )
  root_pos_i = np.stack(
    [np.interp(t_new, t_old, root_pos[:, k]) for k in range(3)],
    axis=1,
  )
  rots = R.from_quat(_fix_quat_wxyz(root_quat)[:, [1, 2, 3, 0]])
  slerp = Slerp(t_old, rots)
  root_quat_i = slerp(t_new).as_quat()[:, [3, 0, 1, 2]]  # back to wxyz
  return _to_tracking(joint_pos, joint_names, root_pos_i, root_quat_i, fps)


def smooth_base_trajectory(
  motion: dict[str, np.ndarray], window: int = 100
) -> dict[str, np.ndarray]:
  """Paper IV-F: Savitzky-Golay smooth base pos + frame-to-frame base rotation.

  Joint trajectories unchanged; body kinematics rebuilt from smoothed root.
  """
  fps = _fps_of(motion)
  joint_names = [str(n) for n in motion["joint_names"]]
  # pelvis is body index matching name
  body_names = [str(n) for n in motion["body_names"]]
  pelvis_i = body_names.index("pelvis")
  root_pos = motion["body_pos_w"][:, pelvis_i].astype(np.float64).copy()
  root_quat = motion["body_quat_w"][:, pelvis_i].astype(np.float64).copy()
  root_quat = _fix_quat_wxyz(root_quat)

  w = int(window)
  if w % 2 == 0:
    w += 1
  w = min(w, len(root_pos) if len(root_pos) % 2 == 1 else len(root_pos) - 1)
  w = max(w, 5)
  poly = 3 if w > 5 else 2

  for k in range(3):
    root_pos[:, k] = savgol_filter(root_pos[:, k], w, poly)

  # Smooth incremental rotation in so(3), then integrate.
  rot = R.from_quat(root_quat[:, [1, 2, 3, 0]])
  drot = (rot[1:] * rot[:-1].inv()).as_rotvec()
  drot_s = np.stack(
    [savgol_filter(drot[:, k], min(w, len(drot) if len(drot) % 2 else len(drot) - 1), poly)
     for k in range(3)],
    axis=1,
  )
  # Rebuild absolute orientation from first frame + smoothed deltas.
  out_quat = [rot[0]]
  for i in range(len(drot_s)):
    out_quat.append(R.from_rotvec(drot_s[i]) * out_quat[-1])
  root_quat_s = R.concatenate(out_quat).as_quat()[:, [3, 0, 1, 2]]

  return _to_tracking(
    motion["joint_pos"].astype(np.float64),
    joint_names,
    root_pos,
    root_quat_s,
    fps,
  )


def planar_vel_stats(motion: dict[str, np.ndarray]) -> dict[str, float]:
  body_names = [str(n) for n in motion["body_names"]]
  i = body_names.index("pelvis")
  quat = motion["body_quat_w"][:, i]
  lin = motion["body_lin_vel_w"][:, i]
  ang = motion["body_ang_vel_w"][:, i]
  rot = R.from_quat(quat[:, [1, 2, 3, 0]])
  yaw = rot.as_euler("xyz")[:, 2]
  c, s = np.cos(yaw), np.sin(yaw)
  vx = c * lin[:, 0] + s * lin[:, 1]
  vy = -s * lin[:, 0] + c * lin[:, 1]
  wz = ang[:, 2]
  fps = _fps_of(motion)
  return {
    "duration_s": float(len(vx) / fps),
    "vx_min": float(vx.min()),
    "vx_max": float(vx.max()),
    "vx_mean": float(vx.mean()),
    "frac_vx_neg": float((vx < -0.05).mean()),
    "vy_abs_mean": float(np.abs(vy).mean()),
    "wz_abs_mean": float(np.abs(wz).mean()),
  }


def build_role(role: str, meta: dict, out_dir: Path) -> dict:
  src = AMP_DIR / meta["source"]
  if not src.exists():
    raise FileNotFoundError(src)
  kind = meta["kind"]
  print(f"\n=== {role} ({kind}) from {src.name} ===")
  if kind == "direct":
    motion = convert_direct(src)
  elif kind == "time_reverse":
    motion = convert_time_reverse(src)
  elif kind == "time_stretch":
    motion = convert_time_stretch(src, float(meta["stretch"]))
  else:
    raise ValueError(kind)

  out_path = out_dir / f"{role}_tracking.npz"
  save(out_path, motion)
  stats = planar_vel_stats(motion)
  print(f"[INFO] wrote {out_path}  stats={stats}")
  return {
    "role": role,
    "file": str(out_path.relative_to(_REPO)),
    "source_amp": str(src.relative_to(_REPO)),
    "kind": kind,
    "notes": meta["notes"],
    "stats": stats,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--also-smoothed",
    action="store_true",
    help="Also write random_walk_smoothed_tracking.npz (paper IV-F, window=100).",
  )
  parser.add_argument(
    "--also-mirror",
    action="store_true",
    help="Also convert walk_mirror.npz → random_walk_mirror_tracking.npz (paper B1).",
  )
  parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
  args = parser.parse_args()
  out_dir: Path = args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  entries = []
  for role, meta in PAPER_ROLES.items():
    entries.append(build_role(role, meta, out_dir))

  if args.also_mirror:
    src = AMP_DIR / "walk_mirror.npz"
    print(f"\n=== random_walk_mirror (direct) from {src.name} ===")
    motion = convert_direct(src)
    path = out_dir / "random_walk_mirror_tracking.npz"
    save(path, motion)
    entries.append(
      {
        "role": "random_walk_mirror",
        "file": str(path.relative_to(_REPO)),
        "source_amp": str(src.relative_to(_REPO)),
        "kind": "direct",
        "notes": "Paper ablation B1 left-right mirrored random walk.",
        "stats": planar_vel_stats(motion),
      }
    )

  if args.also_smoothed:
    rw = out_dir / "random_walk_tracking.npz"
    print(f"\n=== random_walk_smoothed (Savitzky-Golay w=100) from {rw.name} ===")
    raw = np.load(rw, allow_pickle=True)
    motion = {k: raw[k] for k in raw.files}
    smoothed = smooth_base_trajectory(motion, window=100)
    path = out_dir / "random_walk_smoothed_tracking.npz"
    save(path, smoothed)
    entries.append(
      {
        "role": "random_walk_smoothed",
        "file": str(path.relative_to(_REPO)),
        "source_amp": str(rw.relative_to(_REPO)),
        "kind": "savgol_base",
        "notes": "Paper IV-F smoothed base (window=100); joints unchanged.",
        "stats": planar_vel_stats(smoothed),
      }
    )

  dataset = {
    "paper": "arXiv:2611.20888 Driving the Style",
    "paper_pipeline": [
      "human mocap (authors; not public)",
      "GMR retarget to robot",
      "per-frame joint + body references; pelvis anchor",
    ],
    "paper_protocol": {
      "train_generalization": "random_walk only",
      "held_out_eval": ["backward", "side", "slow"],
      "in_distribution_motions": ["random_walk", "backward", "side", "slow", "jog"],
      "eval": "drive with held-out clip velocity profiles; measure vs reference joints",
    },
    "wbot_disclaimer": (
      "Original Agile Robots mocap is not released. Clips below follow the same "
      "semantic roles using existing WBOT AMP locomotion data. backward/slow are "
      "documented approximations."
    ),
    "motions": entries,
  }
  yaml_path = out_dir / "dataset.yaml"
  yaml_path.write_text(yaml.dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
  (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
  print(f"\n[INFO] dataset manifest → {yaml_path}")


if __name__ == "__main__":
  main()
