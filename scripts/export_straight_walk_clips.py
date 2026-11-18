#!/usr/bin/env python3
"""Cut the two mid-speed straight walks out of ``walk_moving.npz``.

The source clip is mostly turns. Its two straight segments still yaw at about
-0.01 to -0.02 rad/s and crab sideways (2–4 cm/m). This export:

1. Keeps those two segments only.
2. Removes heading drift slower than about one second, and the mean
   body-frame lateral velocity. Step sway stays. Joint angles are unchanged.
3. Writes a sagittal mirror of each straightened segment, including key bodies.

Original ``walk_moving*.npz`` / ``walk*.npz`` files are left in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mirror_amp_motion_npz as mirror  # noqa: E402

SRC = REPO / "src/assets/amp_motions/wbot_20dof/walk_run_amp/walk_moving.npz"
MIRROR_SRC = SRC.with_name("walk_moving_mirror.npz")
OUT_DIR = SRC.parent


def _yaw(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = quat_wxyz.T
  return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1.T
  w2, x2, y2, z2 = q2.T
  out = np.stack(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    axis=1,
  )
  out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
  return out


def _yaw_quat(delta: np.ndarray) -> np.ndarray:
  half = 0.5 * delta
  return np.stack(
    (np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)),
    axis=1,
  )


def _rot_z(delta: np.ndarray, vec: np.ndarray) -> np.ndarray:
  c = np.cos(delta)
  s = np.sin(delta)
  lead = vec[..., 0].ndim - c.ndim
  if lead > 0:
    c = c.reshape(c.shape + (1,) * lead)
    s = s.reshape(s.shape + (1,) * lead)
  out = np.empty_like(vec)
  out[..., 0] = c * vec[..., 0] - s * vec[..., 1]
  out[..., 1] = s * vec[..., 0] + c * vec[..., 1]
  out[..., 2] = vec[..., 2]
  return out


def _body_frame_vel(quat: np.ndarray, vel_w: np.ndarray) -> np.ndarray:
  rot = mirror._quat_wxyz_to_rotmat(quat)
  return np.einsum("tij,tj->ti", np.transpose(rot, (0, 2, 1)), vel_w)


def _world_from_body(quat: np.ndarray, vel_b: np.ndarray) -> np.ndarray:
  rot = mirror._quat_wxyz_to_rotmat(quat)
  return np.einsum("tij,tj->ti", rot, vel_b)


def straight_segments(data: dict[str, np.ndarray]) -> list[tuple[int, int]]:
  """Frames where the clip is actually walking forward without a turn."""
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  quat = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
  vel = np.asarray(data["body_lin_vel_w"][:, 0], dtype=np.float64)
  yaw = _yaw(quat)
  c, s = np.cos(yaw), np.sin(yaw)
  vx = c * vel[:, 0] + s * vel[:, 1]
  wz = np.gradient(yaw, 1.0 / fps)
  kernel = np.ones(int(fps)) / fps
  vx_s = np.convolve(vx, kernel, mode="same")
  wz_s = np.convolve(wz, kernel, mode="same")
  mask = (np.abs(wz_s) < 0.15) & (vx_s > 0.25) & (vx_s < 0.9)
  segments: list[tuple[int, int]] = []
  start: int | None = None
  for i, flag in enumerate(mask):
    if flag and start is None:
      start = i
    elif not flag and start is not None:
      if i - start >= int(2 * fps):
        segments.append((start, i))
      start = None
  if start is not None and len(mask) - start >= int(2 * fps):
    segments.append((start, len(mask)))
  return segments


def _path_stats(pos: np.ndarray, quat: np.ndarray, vel: np.ndarray, fps: float) -> str:
  yaw = _yaw(quat)
  y0 = yaw[0]
  c0, s0 = np.cos(y0), np.sin(y0)
  delta = pos - pos[0]
  fwd = c0 * delta[:, 0] + s0 * delta[:, 1]
  lat = -s0 * delta[:, 0] + c0 * delta[:, 1]
  vel_b = _body_frame_vel(quat, vel)
  slope = (yaw[-1] - yaw[0]) / max((len(yaw) - 1) / fps, 1e-6)
  rate = lat[-1] / fwd[-1] * 100.0 if abs(fwd[-1]) > 0.3 else float("nan")
  return (
    f"fwd={fwd[-1]:+.2f}m lat={lat[-1]*100:+.2f}cm ({rate:+.2f} cm/m) "
    f"wz={slope:+.4f} rad/s vx={vel_b[:, 0].mean():+.3f} "
    f"vy={vel_b[:, 1].mean()*100:+.2f} cm/s"
  )


def _heading_correction(yaw: np.ndarray, fps: float, keep_window_s: float = 1.0) -> np.ndarray:
  """Yaw rotation that deletes drift slower than one step and keeps the sway."""
  win = max(3, int(round(keep_window_s * fps)))
  if win % 2 == 0:
    win += 1
  kernel = np.ones(win, dtype=np.float64) / win
  pad = win // 2
  smooth = np.convolve(np.pad(yaw, (pad, pad), mode="edge"), kernel, mode="valid")
  sway = yaw - smooth
  target = yaw[0] + (sway - sway[0])
  return target - yaw


def straighten_segment(data: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
  """Drop heading drift slower than a step, and the mean lateral velocity."""
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  sl = slice(start, end)
  quat = np.asarray(data["body_quat_w"][sl, 0], dtype=np.float64)
  pos = np.asarray(data["body_pos_w"][sl, 0], dtype=np.float64)
  lin = np.asarray(data["body_lin_vel_w"][sl, 0], dtype=np.float64)
  ang = np.asarray(data["body_ang_vel_w"][sl, 0], dtype=np.float64)
  key_pos = np.asarray(data["key_body_pos"][sl], dtype=np.float64).reshape(-1, 6, 3)
  key_quat = np.asarray(data["key_body_quat_w"][sl], dtype=np.float64)

  delta = _heading_correction(_yaw(quat), fps)
  delta_rate = np.gradient(delta, 1.0 / fps)

  vel_b = _body_frame_vel(quat, lin)
  vel_b[:, 1] -= vel_b[:, 1].mean()
  lin_level = _world_from_body(quat, vel_b)

  q_delta = _yaw_quat(delta)
  quat_new = _quat_mul(q_delta, quat)
  lin_new = _rot_z(delta, lin_level)
  ang_new = _rot_z(delta, ang)
  ang_new[:, 2] += delta_rate

  pos_new = np.empty_like(pos)
  pos_new[0] = pos[0]
  dt = 1.0 / fps
  for i in range(1, len(pos)):
    pos_new[i, :2] = pos_new[i - 1, :2] + lin_new[i - 1, :2] * dt
  pos_new[:, 2] = pos[:, 2]

  offset = key_pos - pos[:, None, :]
  key_pos_new = pos_new[:, None, :] + _rot_z(delta, offset)
  key_quat_flat = _quat_mul(
    np.repeat(q_delta, key_quat.shape[1], axis=0),
    key_quat.reshape(-1, 4),
  ).reshape(key_quat.shape)

  out = {key: value[sl].copy() if value.ndim else value for key, value in data.items()}
  out["body_pos_w"] = pos_new[:, None, :].astype(np.float32)
  out["body_quat_w"] = quat_new[:, None, :].astype(np.float32)
  out["body_lin_vel_w"] = lin_new[:, None, :].astype(np.float32)
  out["body_ang_vel_w"] = ang_new[:, None, :].astype(np.float32)
  out["key_body_pos"] = key_pos_new.reshape(len(pos), -1).astype(np.float32)
  out["key_body_quat_w"] = key_quat_flat.astype(np.float32)
  out["joint_pos"] = np.asarray(data["joint_pos"][sl], dtype=np.float32)
  out["joint_vel"] = np.asarray(data["joint_vel"][sl], dtype=np.float32)
  out["fps"] = np.asarray(fps, dtype=np.float64)
  out["joint_names"] = data["joint_names"]
  out["key_body_names"] = data["key_body_names"]
  return out


def _swap_index(names: list[str]) -> np.ndarray:
  name_to_idx = {name: i for i, name in enumerate(names)}
  source = np.arange(len(names))
  for name in names:
    if not name.startswith("left_"):
      continue
    peer = "right_" + name[len("left_") :]
    source[name_to_idx[name]] = name_to_idx[peer]
    source[name_to_idx[peer]] = name_to_idx[name]
  return source


def mirror_clip(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
  key_names = [str(x) for x in np.asarray(data["key_body_names"]).tolist()]
  joint_pos, joint_vel = mirror._mirror_joint_value_arrays(
    joint_pos=np.asarray(data["joint_pos"]),
    joint_vel=np.asarray(data["joint_vel"]),
    joint_names=joint_names,
  )
  body_pos, body_quat, body_lin, body_ang = mirror._mirror_bodies_world(
    body_pos_w=np.asarray(data["body_pos_w"]),
    body_quat_w=np.asarray(data["body_quat_w"]),
    body_lin_vel_w=np.asarray(data["body_lin_vel_w"]),
    body_ang_vel_w=np.asarray(data["body_ang_vel_w"]),
  )
  key_pos = np.asarray(data["key_body_pos"], dtype=np.float64).reshape(
    -1, len(key_names), 3
  )
  key_quat = np.asarray(data["key_body_quat_w"], dtype=np.float64)
  key_pos, key_quat, _, _ = mirror._mirror_bodies_world(
    body_pos_w=key_pos,
    body_quat_w=key_quat,
    body_lin_vel_w=None,
    body_ang_vel_w=None,
  )
  order = _swap_index(key_names)
  out = dict(data)
  out["joint_pos"] = joint_pos.astype(np.float32)
  out["joint_vel"] = joint_vel.astype(np.float32)
  out["body_pos_w"] = body_pos.astype(np.float32)
  out["body_quat_w"] = body_quat.astype(np.float32)
  out["body_lin_vel_w"] = body_lin.astype(np.float32)
  out["body_ang_vel_w"] = body_ang.astype(np.float32)
  out["key_body_pos"] = key_pos[:, order].reshape(len(key_pos), -1).astype(np.float32)
  out["key_body_quat_w"] = key_quat[:, order].astype(np.float32)
  return out


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
  dot = np.abs(np.sum(a * b, axis=-1))
  dot = np.clip(dot, 0.0, 1.0)
  return float(np.degrees(2.0 * np.arccos(dot)).max())


def _check_against_existing_mirror(original: dict[str, np.ndarray]) -> None:
  """The mirror of the source clip must match the clip already in the repo."""
  existing = {key: np.load(MIRROR_SRC)[key] for key in np.load(MIRROR_SRC).files}
  got = mirror_clip(original)
  pos_err = np.max(np.abs(got["key_body_pos"] - existing["key_body_pos"]))
  joint_err = np.max(np.abs(got["joint_pos"] - existing["joint_pos"]))
  root_err = np.max(np.abs(got["body_pos_w"] - existing["body_pos_w"]))
  quat_err = _quat_angle(got["key_body_quat_w"], existing["key_body_quat_w"])
  root_quat_err = _quat_angle(got["body_quat_w"], existing["body_quat_w"])
  print(
    f"[check] mirror vs repo: joint {joint_err:.3e}  root {root_err:.3e}  "
    f"key_pos {pos_err:.3e}  key_quat {quat_err:.3e} deg  "
    f"root_quat {root_quat_err:.3e} deg"
  )
  # Key-body world positions on the stock mirror can differ by ~1 mm; joints/root
  # must still match tightly.
  if max(joint_err, root_err) > 1e-4 or pos_err > 2e-3 or max(quat_err, root_quat_err) > 0.2:
    raise RuntimeError("Mirror does not match walk_moving_mirror.npz")


def _save(path: Path, data: dict[str, np.ndarray]) -> None:
  np.savez(path, **data)
  print(f"[ok] {path.name}: {len(data['joint_pos'])} frames")


def main() -> None:
  with np.load(SRC, allow_pickle=True) as raw:
    data = {key: raw[key] for key in raw.files}
  _check_against_existing_mirror(data)
  segments = straight_segments(data)
  if len(segments) != 2:
    raise RuntimeError(f"Expected 2 straight segments, found {segments}")
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  tags = ("a", "b")
  for tag, (start, end) in zip(tags, segments, strict=True):
    print(
      f"[seg {tag}] frames {start}:{end} "
      f"({start / fps:.2f}-{end / fps:.2f}s)"
    )
    print(
      "  before",
      _path_stats(
        data["body_pos_w"][start:end, 0],
        data["body_quat_w"][start:end, 0],
        data["body_lin_vel_w"][start:end, 0],
        fps,
      ),
    )
    straight = straighten_segment(data, start, end)
    print(
      "  after ",
      _path_stats(
        straight["body_pos_w"][:, 0],
        straight["body_quat_w"][:, 0],
        straight["body_lin_vel_w"][:, 0],
        fps,
      ),
    )
    mirrored = mirror_clip(straight)
    print(
      "  mirror",
      _path_stats(
        mirrored["body_pos_w"][:, 0],
        mirrored["body_quat_w"][:, 0],
        mirrored["body_lin_vel_w"][:, 0],
        fps,
      ),
    )
    _save(OUT_DIR / f"walk_straight_{tag}.npz", straight)
    _save(OUT_DIR / f"walk_straight_{tag}_mirror.npz", mirrored)


if __name__ == "__main__":
  main()
