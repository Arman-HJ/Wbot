"""Build the WBOT ``walk`` AMP clip from the walking parts of ``walk_moving``.

``walk_moving`` mixes straight forward/backward walking, side stepping, in-place
turns, a turning walk and a final stand. Side steps / turns / stand duplicate
the dedicated ``side_*`` / ``turn_*`` / ``stand`` clips, so only the straight
walk and the turning walk are kept and concatenated into ``walk.npz``. Both cuts
sit on near-standing frames (joint jump < one frame step); the second segment's
root path is re-anchored (xy + yaw) onto the end of the first. The mirrored clip
is cut on the same frames. Segments are cross-faded over ``BLEND_S`` (smoothstep;
quaternions via normalized lerp) instead of hard-cut, so velocities do not jump.

Usage:
  python scripts/split_amp_walk_segments.py [--dir src/assets/amp_motions/wbot_20dof/walk_run_amp]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# (start s, end s) inside walk_moving: straight fwd/back walk, then turning walk.
SEGMENTS: tuple[tuple[float, float], ...] = ((0.0, 30.5), (87.5, 111.6))
BLEND_S = 0.5

SOURCES = (("walk_moving.npz", "walk.npz"), ("walk_moving_mirror.npz", "walk_mirror.npz"))

_NAME_KEYS = ("joint_names", "key_body_names", "fps")


def _yaw_quat(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    axis=-1,
  )


def _heading(q: np.ndarray) -> float:
  w, x, y, z = q
  return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _reanchor(
  seg: dict[str, np.ndarray], prev: dict[str, np.ndarray], at: int
) -> dict[str, np.ndarray]:
  """Rotate/translate ``seg`` about z so its first root pose sits on ``prev[at]``."""
  yaw = _heading(prev["body_quat_w"][at, 0]) - _heading(seg["body_quat_w"][0, 0])
  c, s = np.cos(yaw), np.sin(yaw)
  rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  pivot = seg["body_pos_w"][0, 0].copy()
  target = prev["body_pos_w"][at, 0]
  shift = np.array([target[0], target[1], pivot[2]]) - pivot

  def place(p: np.ndarray) -> np.ndarray:
    return (p - pivot) @ rot.T + pivot + shift

  out = dict(seg)
  out["body_pos_w"] = place(seg["body_pos_w"])
  key = seg["key_body_pos"].reshape(len(seg["key_body_pos"]), -1, 3)
  out["key_body_pos"] = place(key).reshape(seg["key_body_pos"].shape)
  for k in ("body_lin_vel_w", "body_ang_vel_w"):
    out[k] = seg[k] @ rot.T
  qz = _yaw_quat(yaw)
  out["body_quat_w"] = _quat_mul(qz, seg["body_quat_w"])
  out["key_body_quat_w"] = _quat_mul(qz, seg["key_body_quat_w"])
  return {k: v.astype(seg[k].dtype) for k, v in out.items()}


def _crossfade(
  a: dict[str, np.ndarray], b: dict[str, np.ndarray], n: int
) -> dict[str, np.ndarray]:
  """Overlap the last ``n`` frames of ``a`` with the first ``n`` of ``b``."""
  b = _reanchor(b, a, len(a["joint_pos"]) - n)
  s = (np.arange(n) + 0.5) / n
  w = s * s * (3.0 - 2.0 * s)
  out: dict[str, np.ndarray] = {}
  for k in a:
    ta, hb = a[k][-n:], b[k][:n]
    wk = w.reshape((n,) + (1,) * (ta.ndim - 1))
    if k in ("body_quat_w", "key_body_quat_w"):
      hb = np.where(np.sum(ta * hb, axis=-1, keepdims=True) < 0.0, -hb, hb)
      mix = (1.0 - wk) * ta + wk * hb
      mix /= np.linalg.norm(mix, axis=-1, keepdims=True)
    else:
      mix = (1.0 - wk) * ta + wk * hb
    out[k] = np.concatenate((a[k][:-n], mix.astype(a[k].dtype), b[k][n:]))
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dir", type=Path, default=Path("src/assets/amp_motions/wbot_20dof/walk_run_amp")
  )
  args = parser.parse_args()

  for src_name, out_name in SOURCES:
    with np.load(args.dir / src_name, allow_pickle=False) as f:
      data = {k: f[k] for k in f.files}
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frames = [k for k in data if k not in _NAME_KEYS]
    blend = int(round(BLEND_S * fps))
    out: dict[str, np.ndarray] | None = None
    for t0, t1 in SEGMENTS:
      a, b = int(round(t0 * fps)), int(round(t1 * fps))
      seg = {k: data[k][a:b] for k in frames}
      out = seg if out is None else _crossfade(out, seg, blend)
    assert out is not None
    out.update({k: data[k] for k in _NAME_KEYS if k in data})
    np.savez(args.dir / out_name, **out)
    print(f"{out_name}: {len(out['joint_pos'])} frames ({len(out['joint_pos']) / fps:.1f}s)")


if __name__ == "__main__":
  main()
