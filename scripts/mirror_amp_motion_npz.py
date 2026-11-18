#!/usr/bin/env python3
"""Mirror an mjlab AMP motion ``.npz`` (no runtime symmetric augmentation).

This exports a left-right mirrored copy of the input motion clip by:
1) Mirroring controlled joints using the same left/right pairing and roll/yaw
   sign rules as ``src/tasks/amp_v2/rl/symmetry.py`` / ``src/tasks/velocity_v3/rl/symmetry.py``.
2) Reflecting world-frame body kinematics across the sagittal plane (y -> -y).
   For angular velocities (pseudo-vector), we apply the correct sign pattern.

Notes:
- This script does not update any dataset yaml. If you want to use the mirrored
  clip, change your dataset entry to point to the new ``*_sym.npz``.
- The mirror operation keeps the stored ``joint_names`` / ``body_names`` order.
  Joint values are swapped/re-signed; body kinematics are reflected in world space.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _find_lr_peer(name: str, name_to_idx: dict[str, int]) -> str | None:
  """Return the opposite-side joint name, if present in ``name_to_idx``."""
  if name.startswith("left_"):
    peer = "right_" + name[len("left_") :]
  elif name.startswith("right_"):
    peer = "left_" + name[len("right_") :]
  elif name.endswith("_Left"):
    peer = name[: -len("_Left")] + "_Right"
  elif name.endswith("_Right"):
    peer = name[: -len("_Right")] + "_Left"
  else:
    return None
  return peer if peer in name_to_idx else None


def _is_yaw_or_roll_joint_name(name: str) -> bool:
  """True for roll/yaw DoFs across Isaac / PND naming conventions."""
  if ("_roll_" in name) or ("_yaw_" in name):
    return True
  if name.endswith("_roll_joint") or name.endswith("_yaw_joint"):
    return True

  # PND / camelCase: hipRoll_Left, shoulderYaw_Right, ankleRoll_Left, ...
  stem, _, side = name.rpartition("_")
  if side not in ("Left", "Right"):
    stem = name
  return ("Roll" in stem) or ("Yaw" in stem)


def _mirror_joint_value_arrays(
  *,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray | None,
  joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
  """Mirror joint angles (swap L/R, negate roll/yaw)."""
  n = len(joint_names)
  if joint_pos.shape[-1] != n:
    raise ValueError(f"joint_pos width {joint_pos.shape[-1]} != len(joint_names) {n}")

  name_to_idx = {name: i for i, name in enumerate(joint_names)}
  source_idx = list(range(n))
  sign = np.ones((n,), dtype=np.float32)

  visited: set[str] = set()
  for name in joint_names:
    if name in visited:
      continue
    peer = _find_lr_peer(name, name_to_idx)
    if peer is not None:
      li = name_to_idx[name]
      ri = name_to_idx[peer]
      source_idx[li] = ri
      source_idx[ri] = li
      visited.add(name)
      visited.add(peer)

  for i, name in enumerate(joint_names):
    if _is_yaw_or_roll_joint_name(name):
      sign[i] = -1.0

  src = np.asarray(source_idx, dtype=np.int64)
  joint_pos_m = joint_pos[..., src] * sign
  joint_vel_m = None
  if joint_vel is not None:
    if joint_vel.shape[-1] != n:
      raise ValueError(f"joint_vel width {joint_vel.shape[-1]} != len(joint_names) {n}")
    joint_vel_m = joint_vel[..., src] * sign

  return joint_pos_m, joint_vel_m


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
  """Convert quaternions (wxyz) to rotation matrices.

  Args:
    q_wxyz: (N, 4)

  Returns:
    R: (N, 3, 3)
  """
  w, x, y, z = [q_wxyz[:, i] for i in range(4)]
  # Rotation matrix, using standard convention for unit quaternions.
  r00 = 1.0 - 2.0 * (y * y + z * z)
  r01 = 2.0 * (x * y - z * w)
  r02 = 2.0 * (x * z + y * w)

  r10 = 2.0 * (x * y + z * w)
  r11 = 1.0 - 2.0 * (x * x + z * z)
  r12 = 2.0 * (y * z - x * w)

  r20 = 2.0 * (x * z - y * w)
  r21 = 2.0 * (y * z + x * w)
  r22 = 1.0 - 2.0 * (x * x + y * y)

  R = np.stack(
    [
      np.stack([r00, r01, r02], axis=1),
      np.stack([r10, r11, r12], axis=1),
      np.stack([r20, r21, r22], axis=1),
    ],
    axis=1,
  )
  return R


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
  """Convert rotation matrices to quaternions (wxyz).

  Args:
    R: (N, 3, 3)

  Returns:
    q_wxyz: (N, 4)
  """
  if R.ndim != 3 or R.shape[1:] != (3, 3):
    raise ValueError(f"Expected R shape (N,3,3), got {R.shape}")

  N = R.shape[0]
  q = np.empty((N, 4), dtype=np.float64)

  r00 = R[:, 0, 0]
  r11 = R[:, 1, 1]
  r22 = R[:, 2, 2]
  trace = r00 + r11 + r22

  # Case 1: trace > 0
  mask = trace > 0.0
  if np.any(mask):
    S = np.sqrt(trace[mask] + 1.0) * 2.0  # S = 4*w
    q[mask, 0] = 0.25 * S
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / S

  # Case 2: trace <= 0, pick largest diagonal element
  mask0 = ~mask & (r00 >= r11) & (r00 >= r22)
  if np.any(mask0):
    S = np.sqrt(1.0 + r00[mask0] - r11[mask0] - r22[mask0]) * 2.0  # S = 4*x
    q[mask0, 0] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / S
    q[mask0, 1] = 0.25 * S
    q[mask0, 2] = (R[mask0, 0, 1] + R[mask0, 1, 0]) / S
    q[mask0, 3] = (R[mask0, 0, 2] + R[mask0, 2, 0]) / S

  mask1 = ~mask & ~mask0 & (r11 >= r22)
  if np.any(mask1):
    S = np.sqrt(1.0 + r11[mask1] - r00[mask1] - r22[mask1]) * 2.0  # S = 4*y
    q[mask1, 0] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
    q[mask1, 1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / S
    q[mask1, 2] = 0.25 * S
    q[mask1, 3] = (R[mask1, 1, 2] + R[mask1, 2, 1]) / S

  mask2 = ~mask & ~mask0 & ~mask1
  if np.any(mask2):
    S = np.sqrt(1.0 + r22[mask2] - r00[mask2] - r11[mask2]) * 2.0  # S = 4*z
    q[mask2, 0] = (R[mask2, 1, 0] - R[mask2, 0, 1]) / S
    q[mask2, 1] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S
    q[mask2, 2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / S
    q[mask2, 3] = 0.25 * S

  # Normalize to unit quaternion (avoid numerical drift).
  norm = np.linalg.norm(q, axis=1, keepdims=True)
  q = q / np.maximum(norm, 1e-12)
  # Enforce consistent sign (optional): make w non-negative.
  q = np.where(q[:, :1] < 0.0, -q, q)
  return q


def _mirror_bodies_world(
  *,
  body_pos_w: np.ndarray | None,
  body_quat_w: np.ndarray | None,
  body_lin_vel_w: np.ndarray | None,
  body_ang_vel_w: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
  """Reflect all body kinematics across sagittal plane in world coords."""
  M = np.asarray([1.0, -1.0, 1.0], dtype=np.float64)  # x, y, z reflection

  if body_pos_w is not None:
    body_pos_w = body_pos_w.copy()
    body_pos_w[..., 1] *= -1.0

  if body_lin_vel_w is not None:
    body_lin_vel_w = body_lin_vel_w.copy()
    body_lin_vel_w[..., 1] *= -1.0

  if body_ang_vel_w is not None:
    body_ang_vel_w = body_ang_vel_w.copy()
    # Angular velocity is a pseudo-vector under reflections: v' = det(M)*M*v.
    # det(M) = -1 => signs become [-1, +1, -1]
    body_ang_vel_w[..., 0] *= -1.0
    body_ang_vel_w[..., 2] *= -1.0

  if body_quat_w is not None:
    body_quat_w = body_quat_w.copy()
    T, B, _ = body_quat_w.shape
    q_flat = body_quat_w.reshape(-1, 4).astype(np.float64)

    R = _quat_wxyz_to_rotmat(q_flat)  # (T*B,3,3)
    # R' = M R M, with M=diag(1,-1,1)
    # Since M is diagonal: scale rows and cols.
    R_prime = R * M[None, :, None] * M[None, None, :]
    q_prime = _rotmat_to_quat_wxyz(R_prime)
    body_quat_w = q_prime.reshape(T, B, 4).astype(body_quat_w.dtype, copy=False)

  return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-npz",
    type=Path,
    required=True,
    help="Source motion npz (mjlab AMP clip).",
  )
  parser.add_argument(
    "--output-npz",
    type=Path,
    default=None,
    help="Output mirrored npz. Default: <stem>_sym.npz beside input.",
  )
  parser.add_argument(
    "--require-keys",
    action="store_true",
    help="Fail if optional arrays (joint_vel/body_vel) are missing.",
  )
  args = parser.parse_args()

  in_path: Path = args.input_npz.expanduser().resolve()
  if args.output_npz is None:
    out_path = in_path.with_name(in_path.stem + "_sym.npz")
  else:
    out_path = args.output_npz.expanduser().resolve()

  if not in_path.is_file():
    raise FileNotFoundError(f"Input npz not found: {in_path}")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  with np.load(in_path, allow_pickle=True) as z:
    data = {k: z[k] for k in z.files}

  if "joint_pos" not in data:
    raise ValueError("Input npz missing required key: 'joint_pos'")

  if "joint_names" in data:
    joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
  elif "joints_list" in data:
    joint_names = [str(x) for x in np.asarray(data["joints_list"]).tolist()]
  else:
    raise ValueError("Input npz must include 'joint_names' or 'joints_list'.")

  joint_pos = np.asarray(data["joint_pos"])
  joint_vel = data.get("joint_vel", None)
  if args.require_keys and joint_vel is None:
    raise ValueError("Input npz missing required key: 'joint_vel'")
  if joint_vel is not None:
    joint_vel = np.asarray(joint_vel)

  joint_pos_sym, joint_vel_sym = _mirror_joint_value_arrays(
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    joint_names=joint_names,
  )
  data["joint_pos"] = joint_pos_sym
  if joint_vel_sym is not None:
    data["joint_vel"] = joint_vel_sym
  elif "joint_vel" in data:
    # Keep key but mirror result missing (shouldn't happen with require_keys=False).
    data.pop("joint_vel", None)

  body_pos_w = data.get("body_pos_w", None)
  body_quat_w = data.get("body_quat_w", None)
  body_lin_vel_w = data.get("body_lin_vel_w", None)
  body_ang_vel_w = data.get("body_ang_vel_w", None)

  if args.require_keys and (body_pos_w is None or body_quat_w is None):
    raise ValueError("Input npz missing required body kinematics keys.")

  if body_pos_w is not None:
    body_pos_w = np.asarray(body_pos_w)
  if body_quat_w is not None:
    body_quat_w = np.asarray(body_quat_w)
  if body_lin_vel_w is not None:
    body_lin_vel_w = np.asarray(body_lin_vel_w)
  if body_ang_vel_w is not None:
    body_ang_vel_w = np.asarray(body_ang_vel_w)

  body_pos_w_sym, body_quat_w_sym, body_lin_vel_w_sym, body_ang_vel_w_sym = (
    _mirror_bodies_world(
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
    )
  )

  if body_pos_w_sym is not None:
    data["body_pos_w"] = body_pos_w_sym
  if body_quat_w_sym is not None:
    data["body_quat_w"] = body_quat_w_sym
  if body_lin_vel_w_sym is not None:
    data["body_lin_vel_w"] = body_lin_vel_w_sym
  if body_ang_vel_w_sym is not None:
    data["body_ang_vel_w"] = body_ang_vel_w_sym

  np.savez(out_path, **data)
  print(f"[OK] Mirrored motion: {in_path} -> {out_path}", flush=True)


if __name__ == "__main__":
  main()

