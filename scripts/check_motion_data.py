#!/usr/bin/env python3
"""Sanity-check WBOT 20DoF motion ``.npz`` files against the current robot XML.

Checks per file: joint order, FK consistency of stored body poses, velocity
fields vs. finite differences (incl. all-zero angular velocity), joint jumps /
hard-limit hits, and ground penetration / floating of the collision geoms.

Usage::

  python scripts/check_motion_data.py a.npz b.npz ... [--xml path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_npz_to_tracking import (  # noqa: E402
  DEFAULT_XML,
  expand,
  floor_model as _floor_model,
  ground_clearance as _ground_clearance,
  report_quality,
)


def check(path: Path, xml: Path, floor_model) -> bool:
  print(f"\n===== {path}")
  d = np.load(path, allow_pickle=False)
  fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  names = [str(n) for n in d["joint_names"]]
  jp = np.asarray(d["joint_pos"], np.float64)
  root_pos = np.asarray(d["body_pos_w"][:, 0], np.float64)
  root_quat = np.asarray(d["body_quat_w"][:, 0], np.float64)
  ok = True

  ref = expand(jp, names, root_pos, root_quat, fps, xml)
  model_joints = [str(n) for n in ref["joint_names"]]
  print(f"[order] {len(names)} joints, fps {fps:g}, {len(jp)} frames ({len(jp) / fps:.1f}s)")

  nb = d["body_pos_w"].shape[1]
  if nb > 1:
    stored = [str(n) for n in d["body_names"]] if "body_names" in d.files else list(ref["body_names"])[:nb]
    idx = [list(ref["body_names"]).index(b) for b in stored]
    pos_err = np.abs(d["body_pos_w"] - ref["body_pos_w"][:, idx]).max()
    q = np.abs(np.abs((d["body_quat_w"] * ref["body_quat_w"][:, idx]).sum(-1)) - 1).max()
    print(f"[FK]    {nb} bodies  max pos err {pos_err * 1000:.2f} mm  quat err {q:.1e}")
    ok &= pos_err < 2e-3
  if "key_body_pos" in d.files and "key_body_names" in d.files:
    kn = [str(n) for n in d["key_body_names"]]
    idx = [list(ref["body_names"]).index(b) for b in kn]
    kp = np.asarray(d["key_body_pos"]).reshape(len(jp), len(kn), 3)
    err = np.abs(kp - ref["body_pos_w"][:, idx]).max()
    print(f"[FK]    key bodies {kn}  max pos err {err * 1000:.2f} mm")
    ok &= err < 2e-3

  jv_err = np.abs(np.asarray(d["joint_vel"]) - ref["joint_vel"])[2:-2].mean()
  lin_err = np.abs(np.asarray(d["body_lin_vel_w"][:, 0]) - ref["body_lin_vel_w"][:, 0])[2:-2].mean()
  w = np.asarray(d["body_ang_vel_w"])
  w_ref = ref["body_ang_vel_w"][:, : w.shape[1]]
  ang_err = np.abs(w - w_ref)[2:-2].mean()
  w_mag = np.linalg.norm(w_ref, axis=-1).mean()
  zero = not np.any(w)
  print(
    f"[vel]   joint_vel err {jv_err:.3f}  root lin err {lin_err:.3f}  "
    f"ang err {ang_err:.3f} (ref |w| {w_mag:.2f}){'  ANGULAR VELOCITY ALL ZERO' if zero else ''}"
  )
  ok &= not zero and ang_err < 0.15 * max(w_mag, 0.1) and jv_err < 0.2

  ok &= report_quality(jp, model_joints, ref["_joint_range"], fps, w_ref)

  model, floor_id = floor_model
  clr = _ground_clearance(model, floor_id, names, jp, root_pos, root_quat)
  pen = clr < -0.01
  print(
    f"[ground] min clearance {clr.min() * 1000:+.1f} mm  p50 {np.median(clr) * 1000:+.1f} mm  "
    f"frames penetrating >10mm: {pen.sum()} ({pen.mean() * 100:.1f}%)  "
    f"frames floating >30mm: {(clr > 0.03).sum()}"
  )
  ok &= pen.mean() < 0.02
  print(f"[result] {'OK' if ok else 'PROBLEM'}")
  return bool(ok)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("files", type=Path, nargs="+")
  p.add_argument("--xml", type=Path, default=DEFAULT_XML)
  args = p.parse_args()
  floor_model = _floor_model(args.xml)
  results = {f: check(f, args.xml, floor_model) for f in args.files}
  print("\n===== summary")
  for f, r in results.items():
    print(f"  {'OK     ' if r else 'PROBLEM'}  {f}")


if __name__ == "__main__":
  main()
