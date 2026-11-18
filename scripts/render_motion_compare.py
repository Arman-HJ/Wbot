#!/usr/bin/env python3
"""Side-by-side kinematic replay of two tracking ``.npz`` clips (WBOT 20DoF).

Top row: full-body side view. Bottom row: foot close-up. Joint readouts are
overlaid per frame.

Usage::

  python scripts/render_motion_compare.py A.npz B.npz --labels old upright \\
    --out compare.mp4 --slow 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gpu_gl import use_gpu_egl

use_gpu_egl()

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[1]
_XML = _REPO / "src/assets/robots/wbot/xmls/wbot_20dof.xml"
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_SHOW = ("left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_knee_joint", "right_knee_joint")


def _model(w: int, h: int) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(_XML))
  spec.visual.global_.offwidth = max(w, 640)
  spec.visual.global_.offheight = max(h, 480)
  world = spec.worldbody
  world.add_light(pos=(2.0, -2.0, 3.0), dir=(-0.5, 0.5, -1.0))
  world.add_light(pos=(-2.0, 2.0, 3.0), dir=(0.5, -0.5, -1.0))
  floor = world.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = (5.0, 5.0, 0.01)
  floor.rgba = (0.35, 0.35, 0.38, 1.0)
  return spec.compile()


def _qpos_map(model: mujoco.MjModel) -> dict[str, int]:
  out = {}
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name:
      out[name.split("/")[-1]] = int(model.jnt_qposadr[j])
  return out


def _cam(distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
  return cam


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("a", type=Path)
  p.add_argument("b", type=Path)
  p.add_argument("--labels", nargs=2, default=("A", "B"))
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--slow", type=float, default=1.0, help="Playback slowdown factor")
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--height", type=int, default=480)
  args = p.parse_args()

  W, H = args.width, args.height
  model = _model(W, H)
  data = mujoco.MjData(model)
  qmap = _qpos_map(model)
  renderer = mujoco.Renderer(model, height=H, width=W)
  font = ImageFont.truetype(_FONT, 20)
  big = ImageFont.truetype(_FONT, 26)

  clips = []
  for path in (args.a, args.b):
    m = np.load(path, allow_pickle=True)
    names = [str(n) for n in m["joint_names"]]
    clips.append(
      dict(
        names=names,
        jp=np.asarray(m["joint_pos"], np.float64),
        root=np.asarray(m["body_pos_w"][:, 0], np.float64),
        quat=np.asarray(m["body_quat_w"][:, 0], np.float64),
        fps=float(np.asarray(m["fps"]).ravel()[0]),
        lim={n: model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _SHOW},
      )
    )
  fps = clips[0]["fps"]
  n = max(c["jp"].shape[0] for c in clips)
  body_cam = _cam(2.4, 90.0, -10.0)
  foot_cam = _cam(1.0, 90.0, -5.0)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(args.out, fps=fps / args.slow, codec="libx264", quality=8)
  for t in range(n):
    cols = []
    for c, label in zip(clips, args.labels):
      k = min(t, c["jp"].shape[0] - 1)
      data.qpos[:] = 0.0
      data.qpos[0:3] = c["root"][k]
      data.qpos[3:7] = c["quat"][k]
      for i, nm in enumerate(c["names"]):
        if nm in qmap:
          data.qpos[qmap[nm]] = c["jp"][k, i]
      mujoco.mj_forward(model, data)
      r = c["root"][k]
      body_cam.lookat[:] = (r[0], r[1], r[2] + 0.1)
      renderer.update_scene(data, camera=body_cam)
      top = Image.fromarray(renderer.render().copy())
      foot_cam.lookat[:] = (r[0], r[1], 0.12)
      renderer.update_scene(data, camera=foot_cam)
      bot = Image.fromarray(renderer.render().copy())

      d = ImageDraw.Draw(top)
      d.text((10, 6), f"{label}   frame {k}/{c['jp'].shape[0] - 1}  t={k / fps:.2f}s", fill=(255, 255, 0), font=big)
      d = ImageDraw.Draw(bot)
      for row, nm in enumerate(_SHOW):
        v = c["jp"][k, c["names"].index(nm)]
        lo, hi = c["lim"][nm]
        at_lim = v <= lo + 0.02 or v >= hi - 0.02
        prev = c["jp"][max(k - 1, 0), c["names"].index(nm)]
        jump = abs(v - prev) > 0.3
        color = (255, 60, 60) if (at_lim or jump) else (255, 255, 255)
        tag = (" 限位" if at_lim else "") + (" 跳变" if jump else "")
        d.text((10, 6 + row * 26), f"{nm.replace('_joint', '')}: {v:+.2f}{tag}", fill=color, font=font)
      col = Image.new("RGB", (W, 2 * H))
      col.paste(top, (0, 0))
      col.paste(bot, (0, H))
      cols.append(col)
    frame = Image.new("RGB", (2 * W + 4, 2 * H), (20, 20, 20))
    frame.paste(cols[0], (0, 0))
    frame.paste(cols[1], (W + 4, 0))
    writer.append_data(np.asarray(frame))
  writer.close()
  renderer.close()
  print(f"[video] {args.out}  frames={n}  playback={fps / args.slow:.0f}fps ({args.slow}x slow)")


if __name__ == "__main__":
  main()
