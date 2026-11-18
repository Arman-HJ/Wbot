#!/usr/bin/env python3
"""Expand a motion ``.npz`` (root body + joints, e.g. GMR/AMP export) into a
full-body mjlab tracking ``.npz`` for WBOT 20DoF.

All velocities are rebuilt from the poses: root linear velocity and joint
velocity by central differences, root angular velocity from consecutive
quaternions, then per-body link velocities from MuJoCo ``cvel`` (same
conversion as mjlab ``EntityData.body_link_*_vel_w``). Velocity fields in the
input are ignored.

Usage::

  python scripts/motion_npz_to_tracking.py \\
    src/assets/motions/wbot_20dof/e1_mimic/dance/victory2.npz \\
    --output src/assets/motions/wbot_20dof/e1_mimic/dance/victory2_tracking.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_XML = _REPO / "src/assets/robots/wbot/xmls/wbot_20dof.xml"
# GMR wbot_20dof qpos[7:27] order (same as the MJCF).
GMR_WBOT_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
  "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
  "right_ankle_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
)


def root_ang_vel_world(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
  """World-frame angular velocity of a (T, 4) wxyz quaternion sequence."""
  rot = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
  if len(rot) < 2:
    return np.zeros((len(rot), 3))
  step = (rot[1:] * rot[:-1].inv()).as_rotvec() * fps
  return np.vstack((step[0], 0.5 * (step[:-1] + step[1:]), step[-1]))


def report_quality(
  joint_pos: np.ndarray,
  joint_names: list[str],
  joint_range: np.ndarray,
  fps: float,
  body_ang_vel_w: np.ndarray | None = None,
  jump_rad_s: float = 15.0,
) -> bool:
  """Print joint jumps / hard-limit hits. Returns False on jumps or all-zero
  angular velocity; hard-limit hits are informational (straight knees sit at 0).

  ``jump_rad_s`` is the per-frame change (expressed as a speed) treated as a
  retarget glitch; real human motion stays well below it.
  """
  ok = True
  step = np.abs(np.diff(joint_pos, axis=0)) * fps
  lo, hi = joint_range[:, 0], joint_range[:, 1]
  at_lim = (joint_pos <= lo + 1e-3) | (joint_pos >= hi - 1e-3)
  print(f"[quality] {len(joint_pos)} frames @ {fps:g} fps")
  for j, name in enumerate(joint_names):
    jumps = np.flatnonzero(step[:, j] > jump_rad_s)
    lim = int(at_lim[:, j].sum())
    if len(jumps) or lim:
      ok = ok and not len(jumps)
      first = ", ".join(str(int(f) + 1) for f in jumps[:6])
      print(
        f"[quality]   {name:28s} jumps>{jump_rad_s:g}rad/s: {len(jumps):4d}"
        f"{' (frames ' + first + ('...' if len(jumps) > 6 else '') + ')' if len(jumps) else ''}"
        f"   at hard limit: {lim} frames"
      )
  if body_ang_vel_w is not None:
    w = np.linalg.norm(body_ang_vel_w, axis=-1)
    print(f"[quality]   body angular velocity: mean {w.mean():.2f}  max {w.max():.2f} rad/s")
    if not np.any(w > 0):
      ok = False
      print("[quality]   body angular velocity is all zero")
  print(f"[quality] {'clean' if ok else 'CHECK THE CLIP ABOVE'}")
  return ok


def expand(
  joint_pos: np.ndarray,
  joint_names: list[str],
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  fps: float,
  xml: Path = DEFAULT_XML,
) -> dict[str, np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(xml))
  data = mujoco.MjData(model)
  if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("expected a floating-base model")

  jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
  if min(jid) < 0:
    raise KeyError(f"joints missing from {xml.name}: {[n for n, i in zip(joint_names, jid) if i < 0]}")
  qadr = model.jnt_qposadr[jid]
  vadr = model.jnt_dofadr[jid]
  body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]

  quat = root_quat_wxyz / np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
  for t in range(1, len(quat)):
    if np.dot(quat[t - 1], quat[t]) < 0.0:
      quat[t] = -quat[t]
  dt = 1.0 / fps
  edge = 2 if len(joint_pos) > 2 else 1
  joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=edge)
  root_lin = np.gradient(root_pos, dt, axis=0, edge_order=edge)
  root_ang = root_ang_vel_world(quat, fps)

  n, nb = len(joint_pos), model.nbody - 1
  out = {
    "body_pos_w": np.zeros((n, nb, 3)),
    "body_quat_w": np.zeros((n, nb, 4)),
    "body_lin_vel_w": np.zeros((n, nb, 3)),
    "body_ang_vel_w": np.zeros((n, nb, 3)),
  }
  local = np.zeros(3)
  conj = np.zeros(4)
  for t in range(n):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = quat[t]
    data.qpos[qadr] = joint_pos[t]
    data.qvel[0:3] = root_lin[t]
    # Free-joint angular velocity is expressed in the body frame.
    mujoco.mju_negQuat(conj, quat[t])
    mujoco.mju_rotVecQuat(local, root_ang[t], conj)
    data.qvel[3:6] = local
    data.qvel[vadr] = joint_vel[t]
    mujoco.mj_forward(model, data)
    ang = data.cvel[1:, :3]
    offset = data.subtree_com[1][None, :] - data.xpos[1:]
    out["body_pos_w"][t] = data.xpos[1:]
    out["body_quat_w"][t] = data.xquat[1:]
    out["body_lin_vel_w"][t] = data.cvel[1:, 3:6] - np.cross(ang, offset)
    out["body_ang_vel_w"][t] = ang

  out = {k: v.astype(np.float32) for k, v in out.items()}
  out["joint_pos"] = joint_pos.astype(np.float32)
  out["joint_vel"] = joint_vel.astype(np.float32)
  out["fps"] = np.asarray([fps], dtype=np.float64)
  out["joint_names"] = np.asarray(joint_names)
  out["body_names"] = np.asarray(body_names)
  out["_joint_range"] = model.jnt_range[jid]
  return out


def save(path: Path, motion: dict[str, np.ndarray]) -> None:
  keys = ("fps", "joint_names", "body_names", "joint_pos", "joint_vel", "body_pos_w",
          "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **{k: motion[k] for k in keys})


def floor_model(xml: Path) -> tuple[mujoco.MjModel, int]:
  spec = mujoco.MjSpec.from_file(str(xml))
  floor = spec.worldbody.add_geom()
  floor.name = "_check_floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = (50.0, 50.0, 0.1)
  floor.margin = 0.3
  model = spec.compile()
  return model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "_check_floor")


def ground_clearance(model, floor_id, joint_names, joint_pos, root_pos, root_quat):
  """Per-frame min signed distance (m) between any robot collision geom and the floor."""
  data = mujoco.MjData(model)
  jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
  qadr = model.jnt_qposadr[jid]
  out = np.full(len(joint_pos), np.inf)
  for t in range(len(joint_pos)):
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat[t]
    data.qpos[qadr] = joint_pos[t]
    mujoco.mj_forward(model, data)
    for c in data.contact[: data.ncon]:
      if floor_id in (c.geom1, c.geom2):
        out[t] = min(out[t], c.dist)
  return out


def ground_fix(
  joint_pos: np.ndarray, joint_names: list[str], root_pos: np.ndarray, root_quat_wxyz: np.ndarray,
  fps: float, xml: Path, sigma_s: float = 0.05,
) -> np.ndarray:
  """Smooth per-frame root height offset keeping the lowest collision geom on the floor.

  Only valid for clips that always keep a support on the ground (no flight phase).
  """
  from scipy.ndimage import gaussian_filter1d, maximum_filter1d

  model, fid = floor_model(xml)
  clr = ground_clearance(model, fid, joint_names, joint_pos, root_pos, root_quat_wxyz)
  sigma = sigma_s * fps
  offset = gaussian_filter1d(-clr, sigma, mode="nearest")
  deficit = np.clip(-(clr + offset), 0.0, None)
  lift = gaussian_filter1d(maximum_filter1d(deficit, size=int(6 * sigma) | 1, mode="nearest"), sigma, mode="nearest")
  offset += np.maximum(lift, deficit)
  print(f"[ground] root z offset min {offset.min() * 1000:+.1f} mm  max {offset.max() * 1000:+.1f} mm")
  out = root_pos.copy()
  out[:, 2] += offset
  return out


def resample(
  joint_pos: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, fps: float, out_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Linear joints / root position, slerp root rotation."""
  if np.isclose(fps, out_fps):
    return joint_pos, root_pos, root_quat_wxyz
  t_in = np.arange(len(joint_pos)) / fps
  t_out = np.minimum(np.arange(0.0, t_in[-1] + 1e-9, 1.0 / out_fps), t_in[-1])
  lerp = lambda x: np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])], axis=1)  # noqa: E731
  rot = Slerp(t_in, R.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]]))(t_out).as_quat()
  return lerp(joint_pos), lerp(root_pos), rot[:, [3, 0, 1, 2]]


def load_source(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, float]:
  """(joint_pos, joint_names, root_pos, root_quat_wxyz, fps) from a motion .npz or a GMR .pkl."""
  if path.suffix == ".pkl":
    with path.open("rb") as f:
      src = pickle.load(f)
    quat_xyzw = np.asarray(src["root_rot"], np.float64)
    return (
      np.asarray(src["dof_pos"], np.float64),
      list(GMR_WBOT_JOINTS),
      np.asarray(src["root_pos"], np.float64),
      quat_xyzw[:, [3, 0, 1, 2]],
      float(src["fps"]),
    )
  src = np.load(path, allow_pickle=False)
  return (
    np.asarray(src["joint_pos"], np.float64),
    [str(n) for n in src["joint_names"]],
    np.asarray(src["body_pos_w"][:, 0], np.float64),
    np.asarray(src["body_quat_w"][:, 0], np.float64),
    float(np.asarray(src["fps"]).reshape(-1)[0]),
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("input", type=Path, help="Motion .npz or GMR .pkl (root_pos/root_rot xyzw/dof_pos)")
  p.add_argument("--output", type=Path, default=None, help="Default: <input>_tracking.npz")
  p.add_argument("--xml", type=Path, default=DEFAULT_XML)
  p.add_argument("--fps", type=float, default=None, help="Resample to this rate (default: keep)")
  p.add_argument("--clip-limits", action="store_true", help="Clip joints to the XML hard limits")
  p.add_argument("--ground-fix", action="store_true", help="Keep the lowest geom on the floor per frame (clips without flight phase)")
  args = p.parse_args()

  joint_pos, names, root_pos, root_quat, fps = load_source(args.input)
  if args.fps:
    joint_pos, root_pos, root_quat = resample(joint_pos, root_pos, root_quat, fps, args.fps)
    fps = args.fps
  if args.clip_limits:
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    rng = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]]
    clipped = np.clip(joint_pos, rng[:, 0], rng[:, 1])
    hit = np.abs(clipped - joint_pos) > 1e-6
    for j in np.flatnonzero(hit.any(0)):
      print(f"[clip]  {names[j]:28s} {hit[:, j].sum():4d} frames, max {np.abs(clipped - joint_pos)[:, j].max():.3f} rad")
    joint_pos = clipped
  if args.ground_fix:
    root_pos = ground_fix(joint_pos, names, root_pos, root_quat, fps, args.xml)
  motion = expand(joint_pos, names, root_pos, root_quat, fps, args.xml)
  report_quality(motion["joint_pos"], list(motion["joint_names"]), motion["_joint_range"], fps,
                 motion["body_ang_vel_w"])
  out = args.output or args.input.with_name(f"{args.input.stem}_tracking.npz")
  save(out, motion)
  print(f"[INFO] {len(motion['joint_pos'])} frames, {len(motion['body_names'])} bodies -> {out}")


if __name__ == "__main__":
  main()
