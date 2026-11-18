"""可视化 AgiBot X2 在指定关节姿态下的初始状态。

用法示例：
  # 直接用 HOME_KEYFRAME 中预设的姿态打开 viewer（仅腿+腰+肩pitch 模型）
  python scripts/view_x2_pose.py --keyframe home

  # 全身模型 + 预设 home
  python scripts/view_x2_pose.py --keyframe full_body_home

  # 全身模型 + running 预设
  python scripts/view_x2_pose.py --keyframe full_body_running

  # 在 home 基础上覆盖膝关节角度（弧度，可重复传入）
  python scripts/view_x2_pose.py --keyframe home \
      --joint left_knee_joint=1.2 --joint right_knee_joint=1.2

  # 用 JSON 文件指定整组关节，键是关节名，值是弧度
  python scripts/view_x2_pose.py --keyframe full_body_home --pose-file pose.json

  # 交互模式：viewer 打开后，可在终端输入命令实时更新姿态
  #   left_knee_joint=1.2          单关节
  #   .*ankle_pitch_joint=-0.3     正则一次性设多个
  #   reset                        恢复到当前 keyframe
  #   dump                         打印当前 qpos（可粘回 keyframe）
  #   list                         列出所有关节及其当前角度
  #   q / quit                     退出
  python scripts/view_x2_pose.py --keyframe full_body_home --interactive

可选项：
  --base-z 0.7        覆盖根 body 的 z 高度
  --no-floor          隐藏地面（直接加载机器人 XML，不加 scene 包裹）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

# 让 `from src ...` 可被解析。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.assets.robots.agibot_x2 import x2_constants  # noqa: E402


KEYFRAMES: dict[str, Any] = {
  "home": x2_constants.HOME_KEYFRAME,
  "full_body_home": x2_constants.HOME_KEYFRAME,
  "full_body_running": x2_constants.HOME_KEYFRAME,
}

# 不同 keyframe 默认使用的 scene/xml。
_SCENE_DIR = x2_constants.X2_XML.parent  # .../x2_assets/
_SCENE_FULL = _SCENE_DIR / "scene.xml"
_SCENE_LEGS = _SCENE_DIR / "scene_waist_arms_pitch_17dof.xml"
_SCENE_LEGS_ONLY = _SCENE_DIR / "scene_legs_only.xml"


def _pick_scene(keyframe_name: str) -> Path:
  """根据 keyframe 选一个合适的 scene。"""
  if keyframe_name == "home":
    # home 用 17-DoF scene 更直观。
    if _SCENE_LEGS.exists():
      return _SCENE_LEGS
  return _SCENE_FULL if _SCENE_FULL.exists() else x2_constants.X2_XML


def _parse_joint_overrides(items: list[str]) -> dict[str, float]:
  out: dict[str, float] = {}
  for s in items:
    if "=" not in s:
      raise SystemExit(f"--joint 参数格式错误，需 name=value：{s!r}")
    name, value = s.split("=", 1)
    name = name.strip()
    try:
      out[name] = float(value)
    except ValueError as e:
      raise SystemExit(f"--joint {s!r} 的 value 不是数字：{e}") from e
  return out


def _all_hinge_joint_names(model: mujoco.MjModel) -> list[str]:
  names = []
  for i in range(model.njnt):
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
      n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if n:
        names.append(n)
  return names


def _apply_joint_positions(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_pos: dict[str, float],
  *,
  silent_missing: bool = False,
) -> list[str]:
  """根据 {joint_name: angle} 设置 qpos，支持正则。返回未匹配到的 key 列表。"""
  hinge_names = _all_hinge_joint_names(model)
  unmatched: list[str] = []
  for key, val in joint_pos.items():
    # 先尝试精确匹配。
    if key in hinge_names:
      qpos_idx = _qpos_index_of(model, key)
      data.qpos[qpos_idx] = float(val)
      continue
    # 然后尝试正则。
    try:
      pat = re.compile(key)
    except re.error:
      pat = None
    matched_any = False
    if pat is not None:
      for n in hinge_names:
        if pat.fullmatch(n):
          qpos_idx = _qpos_index_of(model, n)
          data.qpos[qpos_idx] = float(val)
          matched_any = True
    if not matched_any:
      unmatched.append(key)
  if unmatched and not silent_missing:
    print(f"[WARN] 模型中未匹配到的关节键：{unmatched}")
  return unmatched


def _qpos_index_of(model: mujoco.MjModel, joint_name: str) -> int:
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if jid < 0:
    raise KeyError(joint_name)
  return int(model.jnt_qposadr[jid])


def _set_floating_base(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  *,
  pos: tuple[float, float, float] | None,
) -> None:
  """如果模型存在 free joint，设置其 (x, y, z, qw, qx, qy, qz)。"""
  for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      adr = int(model.jnt_qposadr[i])
      if pos is not None:
        data.qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64)
      # 朝向默认单位四元数。
      data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
      return


def _dump_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  print("# === 当前 qpos（可粘贴回 HOME_KEYFRAME.joint_pos）===")
  for n in _all_hinge_joint_names(model):
    idx = _qpos_index_of(model, n)
    print(f'    "{n}": {float(data.qpos[idx]): .4f},')


def _print_joint_list(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  hinge_names = _all_hinge_joint_names(model)
  print(f"# 模型共 {len(hinge_names)} 个 hinge 关节：")
  for n in hinge_names:
    idx = _qpos_index_of(model, n)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    lo, hi = model.jnt_range[jid]
    print(f"  {n:<32s} q={float(data.qpos[idx]): .4f}  range=[{lo: .3f}, {hi: .3f}]")


def _interactive_loop(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  pose_lock: threading.Lock,
  base_keyframe_joint_pos: dict[str, float],
  base_pos: tuple[float, float, float],
  stop_event: threading.Event,
) -> None:
  """从 stdin 读取命令，热更新 qpos。"""
  print()
  print("=== 交互模式 ===")
  print("输入示例：")
  print("  left_knee_joint=1.2")
  print("  .*ankle_pitch_joint=-0.3       # 正则一次设多个")
  print("  reset                          # 回到当前 keyframe")
  print("  dump                           # 打印当前 qpos")
  print("  list                           # 列出所有关节")
  print("  q / quit                       # 退出")
  print()
  while not stop_event.is_set():
    try:
      line = input("pose> ").strip()
    except (EOFError, KeyboardInterrupt):
      stop_event.set()
      break
    if not line:
      continue
    if line in ("q", "quit", "exit"):
      stop_event.set()
      break
    if line == "dump":
      with pose_lock:
        _dump_qpos(model, data)
      continue
    if line == "list":
      with pose_lock:
        _print_joint_list(model, data)
      continue
    if line == "reset":
      with pose_lock:
        _set_floating_base(model, data, pos=base_pos)
        _apply_joint_positions(model, data, base_keyframe_joint_pos, silent_missing=True)
        mujoco.mj_forward(model, data)
      print("[reset] 已恢复到当前 keyframe。")
      continue
    if "=" not in line:
      print("[ERR] 未识别命令；用 name=value 格式，或 reset/dump/list/q。")
      continue
    name, val_s = line.split("=", 1)
    try:
      val = float(val_s)
    except ValueError:
      print(f"[ERR] 值不是数字：{val_s!r}")
      continue
    with pose_lock:
      unmatched = _apply_joint_positions(
        model, data, {name.strip(): val}, silent_missing=True
      )
      if unmatched:
        print(f"[ERR] 没找到匹配的关节：{unmatched[0]!r}")
      else:
        mujoco.mj_forward(model, data)


def main() -> None:
  parser = argparse.ArgumentParser(description="AgiBot X2 姿态可视化工具")
  parser.add_argument(
    "--keyframe",
    choices=sorted(KEYFRAMES.keys()),
    default="full_body_home",
    help="选用的预设 keyframe（默认 full_body_home）",
  )
  parser.add_argument(
    "--joint",
    action="append",
    default=[],
    metavar="NAME=RAD",
    help="覆盖单个关节角度（弧度），可重复，例如 --joint left_knee_joint=1.2",
  )
  parser.add_argument(
    "--pose-file",
    type=Path,
    default=None,
    help='JSON 文件路径，内容形如 {"left_knee_joint": 1.2, ...}',
  )
  parser.add_argument(
    "--base-z",
    type=float,
    default=None,
    help="覆盖根 body 的 z 高度（米），不传则用 keyframe 默认值",
  )
  parser.add_argument(
    "--no-floor",
    action="store_true",
    help="不加载 scene.xml，直接加载机器人 XML（无地面）",
  )
  parser.add_argument(
    "--interactive",
    action="store_true",
    help="启用 stdin 实时调整关节角度",
  )
  args = parser.parse_args()

  keyframe = KEYFRAMES[args.keyframe]

  # 决定加载哪个 XML。
  if args.no_floor:
    xml_path = (
      x2_constants.X2_LEGS_XML
      if args.keyframe == "home"
      else x2_constants.X2_XML
    )
  else:
    xml_path = _pick_scene(args.keyframe)

  print(f"[load] keyframe = {args.keyframe}")
  print(f"[load] xml      = {xml_path}")
  if not xml_path.exists():
    raise SystemExit(f"XML 文件不存在：{xml_path}")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  # 1) 设置 floating base 位置。
  base_pos = (
    (keyframe.pos[0], keyframe.pos[1], float(args.base_z))
    if args.base_z is not None
    else tuple(keyframe.pos)
  )
  _set_floating_base(model, data, pos=base_pos)

  # 2) 套用 keyframe 中的 joint_pos（用 silent，因为 home 的关节集是子集）。
  base_joint_pos = dict(keyframe.joint_pos)
  _apply_joint_positions(model, data, base_joint_pos, silent_missing=True)

  # 3) 套用 pose-file 覆盖。
  if args.pose_file is not None:
    if not args.pose_file.exists():
      raise SystemExit(f"pose-file 不存在：{args.pose_file}")
    with args.pose_file.open() as f:
      extra = json.load(f)
    if not isinstance(extra, dict):
      raise SystemExit("pose-file 顶层必须是 dict[str, float]")
    _apply_joint_positions(model, data, {k: float(v) for k, v in extra.items()})

  # 4) 套用 CLI --joint 覆盖。
  if args.joint:
    _apply_joint_positions(model, data, _parse_joint_overrides(args.joint))

  mujoco.mj_forward(model, data)

  # 启动交互线程（如果需要）。
  pose_lock = threading.Lock()
  stop_event = threading.Event()
  io_thread: threading.Thread | None = None
  if args.interactive:
    io_thread = threading.Thread(
      target=_interactive_loop,
      args=(model, data, pose_lock, base_joint_pos, base_pos, stop_event),
      daemon=True,
    )
    io_thread.start()

  print("[viewer] 已打开。关闭窗口或在交互模式下输入 q 退出。")
  with mujoco.viewer.launch_passive(model, data) as viewer:
    # 让相机看着 pelvis。
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = int(pelvis_id)
      viewer.cam.distance = 2.5
      viewer.cam.azimuth = 200.0
      viewer.cam.elevation = -10.0

    while viewer.is_running() and not stop_event.is_set():
      with pose_lock:
        # 不做动力学积分，让姿态保持静止。
        mujoco.mj_forward(model, data)
        viewer.sync()
      time.sleep(1.0 / 60.0)

  stop_event.set()
  if io_thread is not None:
    # 输入线程是 daemon，进程结束会自动退出；这里不强制 join。
    pass


if __name__ == "__main__":
  main()
