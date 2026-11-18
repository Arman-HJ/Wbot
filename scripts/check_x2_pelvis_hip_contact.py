#!/usr/bin/env python3
"""检查 X2 pelvis 与 hip_roll 接触（保留碰撞抑制内八时用于调参）。

用法（仓库根目录）::

  python scripts/check_x2_pelvis_hip_contact.py
  python scripts/check_x2_pelvis_hip_contact.py --sweep
  python scripts/check_x2_pelvis_hip_contact.py --xml path/to/x2_ultra_simple.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_XML = _REPO / "src/assets/robots/agibot_x2/x2_assets/x2_ultra_simple.xml"

# 与 x2_constants.HOME_KEYFRAME 腿部一致
_DEFAULT_LEG_JOINTS = {
    "left_hip_pitch_joint": -0.2480,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.5303,
    "left_ankle_pitch_joint": -0.2823,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2480,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.5303,
    "right_ankle_pitch_joint": -0.2823,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
}


def _set_default_pose(model: mujoco.MjModel, data: mujoco.MjData, base_z: float) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if name in _DEFAULT_LEG_JOINTS:
            data.qpos[model.jnt_qposadr[i]] = _DEFAULT_LEG_JOINTS[name]


def _pelvis_hip_roll_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str, str, float]]:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lhr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_roll_link")
    rhr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hip_roll_link")
    hits: list[tuple[str, str, str, float]] = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        bodies = {b1, b2}
        side = None
        if pelvis in bodies and lhr in bodies:
            side = "left"
        elif pelvis in bodies and rhr in bodies:
            side = "right"
        if side is None:
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
        hits.append((side, n1, n2, float(c.dist)))
    return hits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=_DEFAULT_XML)
    p.add_argument("--base-z", type=float, default=0.63, help="自由关节高度（与 HOME_KEYFRAME 一致）")
    p.add_argument("--viewer", action="store_true", help="打开 viewer，在 Rendering 里勾选 Contact point / Contact force")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    _set_default_pose(model, data, args.base_z)
    mujoco.mj_forward(model, data)

    # 检查是否仍存在 exclude
    if model.nexclude > 0:
        print(f"[info] 模型仍有 {model.nexclude} 条 contact exclude（见 XML <contact>）")
        for i in range(model.nexclude):
            b1 = int(model.exclude_body1[i])
            b2 = int(model.exclude_body2[i])
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2)
            if "pelvis" in (n1, n2) and "hip_roll" in (n1, n2):
                print(f"  [WARN] pelvis↔hip_roll 仍被排除: {n1} <-> {n2}")

    hits = _pelvis_hip_roll_contacts(model, data)
    print(f"XML: {args.xml}")
    print(f"默认站姿 base_z={args.base_z}, ncon_total={data.ncon}")
    print(f"pelvis ↔ hip_roll 接触数: {len(hits)}")
    if hits:
        for side, n1, n2, dist in hits:
            print(f"  [{side}] {n1} <-> {n2}  dist={dist:.6f} m  ({'穿透' if dist < 0 else '分离/轻触'})")
    else:
        print("  无接触：几何可能仍分离，可微调 pelvis_collision 的 size/pos 或 hip_roll fromto")

    if args.viewer:
        print("\n[viewer] 左栏 Rendering → 勾选 Contact point / Contact force；空格暂停后拖动关节观察")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
