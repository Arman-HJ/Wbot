"""Convert GMR retarget ``.pkl`` clips to mjlab AMP ``.npz`` (single file or batch).

Single file::

  python scripts/pkl_to_npz.py --input-file path/to/clip.pkl --output-file out.npz

Batch (default X2 locomotion stems)::

  python scripts/pkl_to_npz.py --input-dir /path/to/GMR/retargeted_datasets/... \\
      --output-dir src/assets/amp_motions/x2/robot_stand_walk_run

Joint order is checked before conversion (disable with ``--no-check-joint-order``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

_ADAM_SP_ENV_CFG = (
  "src/tasks/velocity_v3/config/adam_sp/env_cfgs.py",
  "adam_sp_velocity_v3_flat_env_cfg",
)

DEFAULT_ROBOT_ENV_CFG = {
  "agibot_x2": (
    "src/tasks/velocity_v3/config/x2/env_cfgs.py",
    "agibot_x2_velocity_v3_flat_env_cfg",
  ),
  "agibot_x2_full": (
    "src/tasks/velocity_v3/config/x2/env_cfgs.py",
    "agibot_x2_velocity_v3_flat_env_cfg",
  ),
  "pnd_adam_sp": _ADAM_SP_ENV_CFG,
  "adam_sp": _ADAM_SP_ENV_CFG,
  "wbot_12dof": (
    "src/tasks/velocity/config/wbot_12dof/env_cfgs.py",
    "wbot_12dof_flat_env_cfg",
  ),
}

DEFAULT_ROBOT_ASSETS_DIR = {
  "agibot_x2": "src/assets/robots/agibot_x2",
  "agibot_x2_full": "src/assets/robots/agibot_x2",
  "pnd_adam_sp": "src/assets/robots/adam_sp",
  "adam_sp": "src/assets/robots/adam_sp",
  "wbot_12dof": "src/assets/robots/wbot",
}

_X2_MOTOR_JOINT_XML = "src/assets/robots/agibot_x2/x2_assets/x2_ultra_simple.xml"
DEFAULT_ROBOT_JOINT_XML = {
  "agibot_x2": _X2_MOTOR_JOINT_XML,
  "agibot_x2_full": _X2_MOTOR_JOINT_XML,
  "wbot_12dof": "src/assets/robots/wbot/xmls/scene.xml",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent.parent
_DEFAULT_INPUT_DIR = (
  _WORKSPACE_ROOT
  / "Motion_retargeting"
  / "GMR"
  / "retargeted_datasets"
  / "agibot_x2"
  / "robot_stand_walk_run"
)
_DEFAULT_OUTPUT_DIR = (
  _REPO_ROOT / "src" / "assets" / "amp_motions" / "x2" / "robot_stand_walk_run"
)
_DEFAULT_STEMS = ("stand-walk", "walk-run", "run-walk", "walk-stand")
_DEFAULT_GMR_XML = (
  _WORKSPACE_ROOT / "Motion_retargeting" / "GMR" / "assets" / "agibot_x2" / "scene.xml"
)
_DEFAULT_GMR_XML_BY_ROBOT = {
  "agibot_x2": _DEFAULT_GMR_XML,
  "agibot_x2_full": _DEFAULT_GMR_XML,
  "wbot_12dof": _REPO_ROOT.parent / "GMR" / "assets" / "wbot_12dof" / "scene.xml",
  "pnd_adam_sp": (
    _WORKSPACE_ROOT
    / "Motion_retargeting"
    / "GMR"
    / "assets"
    / "pnd_adam_sp"
    / "scene_adam_sp.xml"
  ),
  "adam_sp": (
    _WORKSPACE_ROOT
    / "Motion_retargeting"
    / "GMR"
    / "assets"
    / "pnd_adam_sp"
    / "scene_adam_sp.xml"
  ),
}


def _default_gmr_xml(robot: str) -> Path:
  return Path(_DEFAULT_GMR_XML_BY_ROBOT.get(robot, _DEFAULT_GMR_XML))


def _resolve_gmr_xml_path(robot: str, gmr_xml: str | Path | None) -> Path:
  """Pick GMR model xml; use robot-specific default when CLI still has X2 default."""
  if gmr_xml is None:
    return _default_gmr_xml(robot)
  path = Path(gmr_xml).expanduser().resolve()
  if robot in _DEFAULT_GMR_XML_BY_ROBOT and path == _DEFAULT_GMR_XML.resolve():
    return _default_gmr_xml(robot)
  return path


def _load_env_cfg_fn(env_cfg_path: str, env_cfg_func: str):
  env_cfg_file = Path(env_cfg_path).expanduser().resolve()
  spec = importlib.util.spec_from_file_location("dynamic_env_cfg_module", env_cfg_file)
  if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load env cfg module: {env_cfg_file}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  if not hasattr(module, env_cfg_func):
    raise AttributeError(f"{env_cfg_file} has no function: {env_cfg_func}")
  return getattr(module, env_cfg_func)


def _extract_joint_names_from_xml(xml_path: Path) -> list[str]:
  """Read motor-driven joint order from a MuJoCo XML file."""
  try:
    root = ET.parse(xml_path).getroot()
  except ET.ParseError:
    return []
  names: list[str] = []
  for tag in (".//actuator/motor", ".//actuator/position"):
    for motor in root.findall(tag):
      joint = motor.get("joint")
      if joint:
        names.append(joint)
  return names


def _load_joint_names_from_assets_dir(
  assets_dir: str,
  preferred_xml: str | None = None,
) -> list[str]:
  """Pick joint names from XMLs under robot assets dir.

  Strategy:
  1) Use ``preferred_xml`` when provided and it declares actuator motors.
  2) Otherwise pick the XML that declares the most `<actuator><motor joint=...>`.
  """
  base = Path(assets_dir).expanduser().resolve()
  if not base.exists():
    raise FileNotFoundError(f"assets dir not found: {base}")

  preferred = Path(preferred_xml).expanduser().resolve() if preferred_xml else None
  if preferred is not None and preferred.exists():
    names = _extract_joint_names_from_xml(preferred)
    if names:
      return names

  best_names: list[str] = []
  for xml_path in base.rglob("*.xml"):
    names = _extract_joint_names_from_xml(xml_path)
    if len(names) > len(best_names):
      best_names = names

  if not best_names:
    raise ValueError(f"No actuator motor joints found under: {base}")
  return best_names


class PklMotionLoader:
  def __init__(
    self,
    pkl_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    dof_columns: list[int] | None = None,
    line_range: tuple[int, int] | None = None,
  ):
    self.pkl_file = pkl_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.dof_columns = dof_columns
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    with open(self.pkl_file, "rb") as f:
      motion_dict = pickle.load(f)

    root_pos = np.asarray(motion_dict["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(motion_dict["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion_dict["dof_pos"], dtype=np.float32)
    if self.dof_columns is not None:
      dof_pos = dof_pos[:, self.dof_columns]

    if self.line_range is not None:
      s, e = self.line_range
      s0 = max(0, s - 1)
      e0 = min(root_pos.shape[0], e)
      root_pos = root_pos[s0:e0]
      root_rot_xyzw = root_rot_xyzw[s0:e0]
      dof_pos = dof_pos[s0:e0]

    if root_pos.shape[0] < 2:
      raise ValueError("Need at least 2 frames in pkl motion.")

    self.motion_base_poss_input = torch.from_numpy(root_pos).to(self.device)
    self.motion_base_rots_input = torch.from_numpy(root_rot_xyzw[:, [3, 0, 1, 2]]).to(self.device)
    self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(self.device)

    self.input_frames = self.motion_base_poss_input.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def _mujoco_qpos_dof_joint_names(gmr_xml: str | Path) -> list[str]:
  """Hinge joint names in ``dof_pos`` column order (GMR ``qpos[7:]`` layout)."""
  model = mj.MjModel.from_xml_path(str(gmr_xml))
  hinges: list[tuple[int, str]] = []
  for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE):
      continue
    jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
    if jname is None:
      continue
    col = int(model.jnt_qposadr[j]) - 7
    if col >= 0:
      hinges.append((col, jname))
  hinges.sort(key=lambda x: x[0])
  names = [n for _, n in hinges]
  if model.nq - 7 != len(names):
    raise ValueError(
      f"gmr_xml {gmr_xml}: expected {model.nq - 7} hinge dofs, parsed {len(names)}"
    )
  return names


def _infer_gmr_dof_columns(gmr_xml: str, joint_names: list[str]) -> list[int]:
  name_to_col = {n: i for i, n in enumerate(_mujoco_qpos_dof_joint_names(gmr_xml))}
  missing = [n for n in joint_names if n not in name_to_col]
  if missing:
    raise KeyError(f"Joints not found in gmr_xml ({gmr_xml}): {missing}")
  return [name_to_col[n] for n in joint_names]


def _resolve_npz_joint_names(
  repo_root: Path,
  robot: str,
  joint_names: list[str] | None,
  joint_names_from_assets_dir: str | None,
  *,
  gmr_xml: str | Path | None = None,
) -> list[str]:
  if joint_names is not None and len(joint_names) > 0:
    return list(joint_names)
  if joint_names_from_assets_dir is None:
    rel_assets_dir = DEFAULT_ROBOT_ASSETS_DIR.get(robot, f"src/assets/robots/{robot}")
    joint_names_from_assets_dir = str(repo_root / rel_assets_dir)
  preferred_joint_xml = DEFAULT_ROBOT_JOINT_XML.get(robot)
  if preferred_joint_xml is not None:
    preferred_joint_xml = str(repo_root / preferred_joint_xml)
  try:
    return _load_joint_names_from_assets_dir(
      joint_names_from_assets_dir,
      preferred_xml=preferred_joint_xml,
    )
  except ValueError:
    gmr_path = Path(gmr_xml).expanduser().resolve() if gmr_xml else _default_gmr_xml(robot)
    if gmr_path.is_file():
      names = _mujoco_qpos_dof_joint_names(str(gmr_path))
      print(
        f"Loaded {len(names)} joint names from GMR xml (no actuator motors in assets): "
        f"{gmr_path.name}"
      )
      return names
    raise


def _load_pkl_joint_metadata(
  pkl_data: dict[str, Any],
  pkl_path: Path,
) -> tuple[list[str] | None, str | None]:
  """Return (joint_names, dof_column_order) from pickle or sibling ``joint_order.json``."""
  raw = pkl_data.get("joint_names")
  if raw is not None:
    names = [str(x) for x in (raw.tolist() if hasattr(raw, "tolist") else raw)]
    order = pkl_data.get("dof_column_order")
    return names, str(order) if order is not None else None

  meta_path = pkl_path.parent / "joint_order.json"
  if meta_path.is_file():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cols = meta.get("dof_pos_columns")
    if cols:
      return [str(x) for x in cols], str(meta.get("dof_column_order", "mujoco_qpos"))
  return None, None


def check_pkl_joint_order(
  input_file: str | Path,
  *,
  robot: str = "agibot_x2_full",
  gmr_xml: str | Path | None = None,
  joint_names: list[str] | None = None,
  joint_names_from_assets_dir: str | None = None,
  repo_root: Path | None = None,
) -> dict[str, Any]:
  """Compare GMR pkl ``dof_pos`` columns vs motor (npz) vs MuJoCo qpos (GMR xml).

  Raises ``ValueError`` if checks fail. Returns a summary dict when ok.
  """
  repo_root = repo_root or Path(__file__).resolve().parents[1]
  pkl_path = Path(input_file).expanduser().resolve()
  if gmr_xml is None:
    gmr_xml = _default_gmr_xml(robot)
  gmr_xml = Path(gmr_xml).expanduser().resolve()

  with pkl_path.open("rb") as f:
    pkl_data = pickle.load(f)

  dof_pos = np.asarray(pkl_data["dof_pos"])
  if dof_pos.ndim != 2:
    raise ValueError(f"{pkl_path}: dof_pos must be 2D, got shape {dof_pos.shape}")

  motor_names = _resolve_npz_joint_names(
    repo_root, robot, joint_names, joint_names_from_assets_dir, gmr_xml=gmr_xml
  )
  gmr_qpos_names = _mujoco_qpos_dof_joint_names(gmr_xml)
  pkl_names, pkl_order = _load_pkl_joint_metadata(pkl_data, pkl_path)

  n_dof = dof_pos.shape[1]
  errors: list[str] = []

  if n_dof != len(motor_names):
    errors.append(
      f"dof_pos width {n_dof} != motor joint count {len(motor_names)}"
    )
  if n_dof != len(gmr_qpos_names):
    errors.append(
      f"dof_pos width {n_dof} != gmr qpos dof count {len(gmr_qpos_names)}"
    )

  if set(motor_names) != set(gmr_qpos_names):
    errors.append("motor joint set != gmr qpos joint set")

  if pkl_names is not None:
    if len(pkl_names) != n_dof:
      errors.append(
        f"joint_names len {len(pkl_names)} != dof_pos width {n_dof}"
      )
    if pkl_names != gmr_qpos_names:
      errors.append(
        "pkl joint_names != GMR mujoco_qpos column order "
        "(re-export GMR pkl or fix joint_order.json)"
      )
    if set(pkl_names) != set(motor_names):
      errors.append("pkl joint_names missing/extra vs motor list")
  else:
    print(
      f"[WARN] {pkl_path.name}: no joint_names in pkl and no joint_order.json; "
      "assuming dof_pos columns follow GMR mujoco_qpos order."
    )

  dof_columns = _infer_gmr_dof_columns(str(gmr_xml), motor_names)
  if pkl_names is not None:
    expected_cols = [pkl_names.index(n) for n in motor_names]
    if dof_columns != expected_cols:
      errors.append(
        "dof column map from gmr_xml disagrees with pkl joint_names "
        f"(first diff index {next((i for i, (a, b) in enumerate(zip(dof_columns, expected_cols)) if a != b), '?')})"
      )

  mismatches = [
    (i, motor_names[i], gmr_qpos_names[i])
    for i in range(min(len(motor_names), len(gmr_qpos_names)))
    if motor_names[i] != gmr_qpos_names[i]
  ]

  print(f"\n=== joint order check: {pkl_path.name} ===")
  print(f"npz joint_names target: motor / actuator ({len(motor_names)} joints)")
  print(f"pkl dof_pos columns:    mujoco_qpos / GMR qpos[7:] ({len(gmr_qpos_names)} joints)")
  if pkl_names is not None:
    print(f"pkl metadata joint_names: {pkl_order or 'present'}")
  print(f"motor == gmr qpos name order at same index: {not mismatches}")
  if mismatches:
    print("Index pairs where motor list != gmr qpos list (expected for X2 at 15–30):")
    for i, mname, qname in mismatches:
      print(f"  {i:2d}  motor={mname:32s}  gmr_qpos_col={qname:32s}")
  if pkl_names is not None and pkl_names == gmr_qpos_names:
    print("pkl joint_names matches gmr_qpos column order.")
  print(f"dof_columns (motor joint -> pkl col): {dof_columns[:5]}...{dof_columns[-3:]}")

  if errors:
    msg = f"Joint order check FAILED for {pkl_path}:\n  - " + "\n  - ".join(errors)
    raise ValueError(msg)

  print("Joint order check PASSED.\n")
  return {
    "pkl_file": str(pkl_path),
    "num_dof": n_dof,
    "motor_names": motor_names,
    "gmr_qpos_names": gmr_qpos_names,
    "pkl_joint_names": pkl_names,
    "dof_columns": dof_columns,
    "motor_qpos_index_mismatches": len(mismatches),
  }


def _find_mujoco_joint_id(model: mj.MjModel, joint_name: str, entity_name: str = "robot") -> int:
  for candidate in (joint_name, f"{entity_name}/{joint_name}"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, candidate)
    if jid >= 0:
      return jid

  matches = []
  suffix = f"/{joint_name}"
  for jid in range(model.njnt):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
    if name is not None and name.endswith(suffix):
      matches.append(jid)

  if len(matches) == 1:
    return matches[0]
  if len(matches) > 1:
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) for jid in matches]
    raise KeyError(f"Multiple joints match '{joint_name}' in preview MuJoCo model: {names}")
  raise KeyError(f"Joint not found in preview MuJoCo model: {joint_name}")


def _export_base_lin_vel_curve(
  output_path: str,
  log: dict[str, Any],
  base_body_name: str,
  curve_output_dir: str | None = None,
) -> None:
  body_names = [str(x) for x in log["body_names"].tolist()]
  if base_body_name not in body_names:
    raise ValueError(
      f"base_body_name '{base_body_name}' not found in body_names: {body_names}"
    )
  body_index = body_names.index(base_body_name)

  body_lin_vel_w = log["body_lin_vel_w"]  # [T, B, 3]
  base_lin_vel = body_lin_vel_w[:, body_index, :]
  speed = np.linalg.norm(base_lin_vel, axis=1)
  fps = float(log["fps"][0])
  t = np.arange(base_lin_vel.shape[0], dtype=np.float32) / fps

  out_dir = (
    Path(curve_output_dir).expanduser().resolve()
    if curve_output_dir
    else Path(output_path).expanduser().resolve().parent
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  stem = Path(output_path).stem
  csv_path = out_dir / f"{stem}_base_lin_vel.csv"
  png_path = out_dir / f"{stem}_base_lin_vel.png"

  csv_data = np.column_stack([t, base_lin_vel, speed])
  np.savetxt(
    csv_path,
    csv_data,
    delimiter=",",
    header="time_s,vx,vy,vz,speed",
    comments="",
  )

  plt.figure(figsize=(12, 6))
  plt.plot(t, base_lin_vel[:, 0], label="vx")
  plt.plot(t, base_lin_vel[:, 1], label="vy")
  plt.plot(t, base_lin_vel[:, 2], label="vz")
  plt.plot(t, speed, label="speed", linewidth=2.0)
  plt.xlabel("Time (s)")
  plt.ylabel("Linear velocity (m/s)")
  plt.title(f"Base linear velocity - {base_body_name}")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(png_path, dpi=150)
  plt.close()

  print(f"Saved base linear velocity csv: {csv_path}", flush=True)
  print(f"Saved base linear velocity plot: {png_path}", flush=True)


def convert_pkl_to_npz(
  input_file: str,
  output_name: str = "",
  robot: str = "agibot_x2_full",
  env_cfg_path: str | None = None,
  env_cfg_func: str | None = None,
  joint_names: list[str] | None = None,
  joint_names_from_assets_dir: str | None = None,
  gmr_xml: str = "/home/songhat/code_workspace/Motion_retargeting/GMR/assets/agibot_x2/scene.xml",
  output_dir: str = "outputs",
  output_file: str | None = None,
  input_fps: int | None = None,
  output_fps: int = 50,
  device: str = "cuda:0",
  line_range: tuple[int, int] | None = None,
  preview: bool = False,
  preview_rate_limit: bool = True,
  export_base_lin_vel_curve: bool = False,
  base_body_name: str = "pelvis",
  curve_output_dir: str | None = None,
  check_joint_order: bool = True,
):
  """Convert retarget pkl to npz using task Scene/Simulation pipeline."""
  repo_root = Path(__file__).resolve().parents[1]
  gmr_xml = str(_resolve_gmr_xml_path(robot, gmr_xml))

  if env_cfg_path is None or env_cfg_func is None:
    if robot not in DEFAULT_ROBOT_ENV_CFG:
      raise ValueError(
        f"Unknown robot '{robot}'. Please provide --env-cfg-path and --env-cfg-func explicitly."
      )
    rel_path, default_func = DEFAULT_ROBOT_ENV_CFG[robot]
    env_cfg_path = str(repo_root / rel_path)
    env_cfg_func = default_func

  if output_file is not None and output_file != "":
    output_path = str(Path(output_file).expanduser().resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  else:
    if output_name == "":
      raise ValueError("Either --output-file or --output-name must be provided.")
    os.makedirs(output_dir, exist_ok=True)
    if not output_name.endswith(".npz"):
      output_name += ".npz"
    output_path = os.path.join(output_dir, output_name)

  with open(input_file, "rb") as f:
    pkl_data = pickle.load(f)
  if check_joint_order:
    check_pkl_joint_order(
      input_file,
      robot=robot,
      gmr_xml=gmr_xml,
      joint_names=joint_names,
      joint_names_from_assets_dir=joint_names_from_assets_dir,
      repo_root=repo_root,
    )
  fps_from_pkl = int(pkl_data.get("fps", 30))
  fps = fps_from_pkl if input_fps is None else int(input_fps)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / float(output_fps)
  env_cfg_fn = _load_env_cfg_fn(env_cfg_path, env_cfg_func)
  scene = Scene(env_cfg_fn().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  resolved_joint_names = joint_names
  if resolved_joint_names is None or len(resolved_joint_names) == 0:
    resolved_joint_names = _resolve_npz_joint_names(
      repo_root,
      robot,
      joint_names,
      joint_names_from_assets_dir,
      gmr_xml=gmr_xml,
    )
    print(f"Resolved {len(resolved_joint_names)} npz joint_names.")

  dof_columns = _infer_gmr_dof_columns(gmr_xml, resolved_joint_names)
  print(f"Mapped {len(dof_columns)} pkl dof columns from gmr_xml.")

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(resolved_joint_names, preserve_order=True)[0]
  preview_data = mj.MjData(sim.mj_model) if preview else None
  preview_joint_qposadr = []
  preview_joint_qveladr = []
  if preview:
    for name in resolved_joint_names:
      jid = _find_mujoco_joint_id(sim.mj_model, name)
      preview_joint_qposadr.append(int(sim.mj_model.jnt_qposadr[jid]))
      preview_joint_qveladr.append(int(sim.mj_model.jnt_dofadr[jid]))

  motion = PklMotionLoader(
    pkl_file=input_file,
    input_fps=fps,
    output_fps=output_fps,
    device=sim.device,
    dof_columns=dof_columns,
    line_range=line_range,
  )

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_names": np.array(resolved_joint_names),
    "body_names": np.array(robot.body_names),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False
  step_dt = float(sim.mj_model.opt.timestep)

  def run_loop(viewer=None):
    nonlocal file_saved
    scene.reset()
    pbar = tqdm(total=motion.output_frames, desc="Converting", unit="frame")
    while not file_saved:
      if viewer is not None and not viewer.is_running():
        raise RuntimeError("Viewer closed before conversion finished.")
      t0 = time.time()
      (
        (
          motion_base_pos,
          motion_base_rot,
          motion_base_lin_vel,
          motion_base_ang_vel,
          motion_dof_pos,
          motion_dof_vel,
        ),
        reset_flag,
      ) = motion.get_next_state()

      root_states = robot.data.default_root_state.clone()
      root_states[:, 0:3] = motion_base_pos
      root_states[:, :2] += scene.env_origins[:, :2]
      root_states[:, 3:7] = motion_base_rot
      root_states[:, 7:10] = motion_base_lin_vel
      root_states[:, 10:] = motion_base_ang_vel
      robot.write_root_state_to_sim(root_states)

      joint_pos = robot.data.default_joint_pos.clone()
      joint_vel = robot.data.default_joint_vel.clone()
      joint_pos[:, robot_joint_indexes] = motion_dof_pos
      joint_vel[:, robot_joint_indexes] = motion_dof_vel
      robot.write_joint_state_to_sim(joint_pos, joint_vel)

      sim.forward()
      scene.update(sim.mj_model.opt.timestep)

      # Column i must match joint_names[i] (motor / resolved_joint_names order).
      log["joint_pos"].append(
        robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy()
      )
      log["joint_vel"].append(
        robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy()
      )
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
      log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())
      pbar.update(1)

      if viewer is not None and preview_data is not None:
        preview_data.qpos[:] = 0.0
        preview_data.qvel[:] = 0.0
        preview_data.qpos[0:3] = root_states[0, 0:3].detach().cpu().numpy()
        preview_data.qpos[3:7] = root_states[0, 3:7].detach().cpu().numpy()
        preview_data.qvel[0:3] = root_states[0, 7:10].detach().cpu().numpy()
        preview_data.qvel[3:6] = root_states[0, 10:13].detach().cpu().numpy()
        dof_np = motion_dof_pos[0].detach().cpu().numpy()
        dof_vel_np = motion_dof_vel[0].detach().cpu().numpy()
        for qpos_adr, qpos_value in zip(preview_joint_qposadr, dof_np, strict=True):
          preview_data.qpos[qpos_adr] = qpos_value
        for qvel_adr, qvel_value in zip(preview_joint_qveladr, dof_vel_np, strict=True):
          preview_data.qvel[qvel_adr] = qvel_value
        mj.mj_forward(sim.mj_model, preview_data)
        viewer.sync()
      if preview_rate_limit:
        elapsed = time.time() - t0
        if elapsed < step_dt:
          time.sleep(step_dt - elapsed)

      if reset_flag:
        file_saved = True

    pbar.close()

  if preview:
    assert preview_data is not None
    with mujoco.viewer.launch_passive(sim.mj_model, preview_data) as viewer:
      viewer.sync()
      run_loop(viewer=viewer)
  else:
    run_loop(viewer=None)

  for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
    log[k] = np.stack(log[k], axis=0)

  np.savez(output_path, **log)
  print(f"Saved npz to: {output_path}", flush=True)
  if export_base_lin_vel_curve:
    _export_base_lin_vel_curve(
      output_path=output_path,
      log=log,
      base_body_name=base_body_name,
      curve_output_dir=curve_output_dir,
    )
  if preview:
    # mujoco.viewer can segfault during Python finalization after Warp-backed simulation.
    # The output is already flushed and saved, so exit without running fragile destructors.
    os._exit(0)


def batch_pkl_to_npz(
  input_dir: Path = _DEFAULT_INPUT_DIR,
  output_dir: Path = _DEFAULT_OUTPUT_DIR,
  stems: tuple[str, ...] = _DEFAULT_STEMS,
  robot: str = "agibot_x2_full",
  suffix: str = "_agibot_x2",
  skip_existing: bool = False,
  check_joint_order: bool = True,
  gmr_xml: Path = _DEFAULT_GMR_XML,
  output_fps: int = 50,
  device: str = "cuda:0",
) -> None:
  """Convert ``{stem}{suffix}.pkl`` in ``input_dir`` to ``.npz`` in ``output_dir``."""
  input_dir = input_dir.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()
  gmr_xml_path = _resolve_gmr_xml_path(robot, gmr_xml)
  if not input_dir.is_dir():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")
  if check_joint_order and not gmr_xml_path.is_file():
    raise FileNotFoundError(f"GMR model xml not found: {gmr_xml_path}")

  output_dir.mkdir(parents=True, exist_ok=True)
  gmr_xml_str = str(gmr_xml_path)

  for stem in stems:
    name = f"{stem}{suffix}"
    input_path = input_dir / f"{name}.pkl"
    output_path = output_dir / f"{name}.npz"

    if not input_path.is_file():
      raise FileNotFoundError(f"Missing input pkl: {input_path}")
    if skip_existing and output_path.is_file():
      print(f"[SKIP] {output_path} already exists", flush=True)
      continue

    if check_joint_order:
      check_pkl_joint_order(
        input_path,
        robot=robot,
        gmr_xml=gmr_xml,
        repo_root=_REPO_ROOT,
      )

    print(f"[INFO] {input_path} -> {output_path}", flush=True)
    convert_pkl_to_npz(
      input_file=str(input_path),
      output_file=str(output_path),
      robot=robot,
      gmr_xml=gmr_xml_str,
      output_fps=output_fps,
      device=device,
      check_joint_order=False,
    )


def main(
  input_file: str = "",
  input_dir: Path | None = None,
  output_dir: Path = _DEFAULT_OUTPUT_DIR,
  stems: tuple[str, ...] = _DEFAULT_STEMS,
  suffix: str = "_agibot_x2",
  skip_existing: bool = False,
  check_joint_order: bool = True,
  output_name: str = "",
  robot: str = "agibot_x2_full",
  env_cfg_path: str | None = None,
  env_cfg_func: str | None = None,
  joint_names: list[str] | None = None,
  joint_names_from_assets_dir: str | None = None,
  gmr_xml: str | Path = _DEFAULT_GMR_XML,
  output_file: str | None = None,
  input_fps: int | None = None,
  output_fps: int = 50,
  device: str = "cuda:0",
  line_range: tuple[int, int] | None = None,
  preview: bool = False,
  preview_rate_limit: bool = True,
  export_base_lin_vel_curve: bool = False,
  base_body_name: str = "pelvis",
  curve_output_dir: str | None = None,
) -> None:
  """CLI entry: use ``--input-dir`` for batch, or ``--input-file`` for a single clip."""
  if input_dir is not None:
    batch_pkl_to_npz(
      input_dir=input_dir,
      output_dir=output_dir,
      stems=stems,
      robot=robot,
      suffix=suffix,
      skip_existing=skip_existing,
      check_joint_order=check_joint_order,
      gmr_xml=Path(gmr_xml),
      output_fps=output_fps,
      device=device,
    )
    return

  if not input_file:
    raise ValueError("Provide --input-file (single) or --input-dir (batch).")

  gmr_xml = _resolve_gmr_xml_path(robot, gmr_xml)

  convert_pkl_to_npz(
    input_file=input_file,
    output_name=output_name,
    robot=robot,
    env_cfg_path=env_cfg_path,
    env_cfg_func=env_cfg_func,
    joint_names=joint_names,
    joint_names_from_assets_dir=joint_names_from_assets_dir,
    gmr_xml=str(gmr_xml),
    output_dir=str(output_dir),
    output_file=output_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    line_range=line_range,
    preview=preview,
    preview_rate_limit=preview_rate_limit,
    export_base_lin_vel_curve=export_base_lin_vel_curve,
    base_body_name=base_body_name,
    curve_output_dir=curve_output_dir,
    check_joint_order=check_joint_order,
  )


if __name__ == "__main__":
  tyro.cli(main)

