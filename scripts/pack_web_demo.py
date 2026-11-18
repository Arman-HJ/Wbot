#!/usr/bin/env python3
"""Pack training runs into browser WASM play assets — task-aware (Viser parity).

Mirrors scripts/play.py / play_amp.py routing:
  - AMP-Walk-*  → play_profile=amp_twist  (UniformVelocityCommand GUI)
  - *Kick* / *Tracking-* → play_profile=kick  (motion pack + push panel)
  - Unitree-H2-* → play_profile=velocity  (twist GUI + gait phase obs)
  - Davis-Shoot-* → play_profile=davis  (proprio+depth dual-input ONNX)
  - other tasks → rejected until a matching web profile exists

Robot MJCF is chosen from the run's controlled joint count / task_id
(WBOT 12-DoF / 20-DoF, Unitree H2 29-DoF), NOT a single hardcoded XML.

Usage:
  # One run → Pack dropdown only lists that run (default replaces index)
  python scripts/pack_web_demo.py --run logs/rsl_rl/.../my_run
  # Multiple runs for one task family (still replaces index with just these)
  python scripts/pack_web_demo.py --all-logs --task AMP-Walk-Run-Wbot-20DOF --limit 5
  # Merge into existing index without pruning:
  python scripts/pack_web_demo.py --run logs/rsl_rl/.../my_run --no-replace-index
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = ROOT / "logs" / "rsl_rl"
DEFAULT_OUT = ROOT / "web_demo" / "public" / "packs"
SHARED_ROBOTS = ROOT / "web_demo" / "public" / "shared" / "robots"
ROBOT_XML_DIR = ROOT / "src" / "assets" / "robots" / "wbot" / "xmls"
ROBOT_MESHES = ROBOT_XML_DIR / "meshes"
H2_XML = ROOT / "src" / "assets" / "robots" / "unitree_h2" / "xmls" / "h2.xml"
H2_MESHES = H2_XML.parent / "assets"

_ROBOT_MESH_SRC = {"wbot": ROBOT_MESHES, "h2": H2_MESHES}

# H2 BuiltinPositionActuatorCfg uses armature=0.01 on every joint (h2_constants).
_H2_ARMATURE = 0.01

# Match training: sim dt 0.005, decimation 4 → 50 Hz policy.
DEFAULT_SIM_DT = 0.005
DEFAULT_DECIMATION = 4

# Reflected motor inertia from wbot_20dof_constants (training injects these via
# BuiltinPositionActuatorCfg.armature). Raw MJCF has none → without this the
# browser dynamics are far too agile and the policy cannot balance.
_ARMATURE_BY_PATTERN: list[tuple[re.Pattern[str], float]] = [
  (re.compile(r".*_ankle_roll_joint"), 5607.889 / 1e6),
  (re.compile(r".*_hip_yaw_joint"), 24222.8413745986 / 1e6),
  (re.compile(r".*_ankle_pitch_joint"), 24222.8413745986 / 1e6),
  (re.compile(r".*_hip_pitch_joint"), 32980.2813208958 / 1e6),
  (re.compile(r".*_hip_roll_joint"), 32980.2813208958 / 1e6),
  (re.compile(r".*_knee_joint"), 32980.2813208958 / 1e6),
  (re.compile(r".*_shoulder_pitch_joint"), 0.001),
  (re.compile(r".*_shoulder_roll_joint"), 0.001),
  (re.compile(r".*_shoulder_yaw_joint"), 0.0007),
  (re.compile(r".*_elbow_joint"), 0.0007),
  # DAVIS 2-DoF neck (same arm-class motor as shoulder_yaw).
  (re.compile(r"neck_yaw_joint"), 0.0007),
  (re.compile(r"neck_pitch_joint"), 0.0007),
]


def _armature_for_joint(name: str) -> float | None:
  for pattern, value in _ARMATURE_BY_PATTERN:
    if pattern.fullmatch(name):
      return float(value)
  return None


def _inject_joint_attr(
  inner: str,
  joint_names: list[str],
  attr: str,
  lookup,
  *,
  warn_label: str,
) -> str:
  """Set attr=\"…\" on matching joint tags (overwrite if already present)."""
  for name in joint_names:
    value = lookup(name)
    if value is None:
      continue
    pattern = rf'(<joint\b[^>]*\bname="{re.escape(name)}"[^>]*?)(/?>)'

    def _repl(m: re.Match[str], v: float = value) -> str:
      tag = m.group(1)
      if f"{attr}=" in tag:
        tag = re.sub(rf'\s*{re.escape(attr)}="[^"]*"', "", tag)
      return f'{tag} {attr}="{v}"{m.group(2)}'

    inner, n = re.subn(pattern, _repl, inner, count=1)
    if n == 0:
      print(f"[WARN] joint tag not found for {warn_label} inject: {name}", file=sys.stderr)
  return inner


def _inject_joint_armature(
  inner: str, joint_names: list[str], family: str = "wbot"
) -> str:
  """Add armature=… on controlled joints so PD bandwidth matches training."""
  lookup = (lambda _name: _H2_ARMATURE) if family == "h2" else _armature_for_joint
  return _inject_joint_attr(
    inner, joint_names, "armature", lookup, warn_label="armature"
  )


_H2_FOOT_GEOM = re.compile(r"(left|right)_foot[1-7]_collision")


def _apply_h2_collision(inner: str) -> str:
  """Match h2_constants.FULL_COLLISION: feet condim=3/priority=1/friction=0.6, rest condim=1."""

  def _set(tag: str, attr: str, value: str) -> str:
    tag = re.sub(rf'\s{attr}="[^"]*"', "", tag)
    return f'{tag} {attr}="{value}"'

  def _repl(m: re.Match[str]) -> str:
    tag, end = m.group(1), m.group(2)
    name_m = re.search(r'\bname="([^"]+_collision)"', tag)
    if name_m is None:
      return m.group(0)
    if _H2_FOOT_GEOM.fullmatch(name_m.group(1)):
      tag = _set(tag, "condim", "3")
      tag = _set(tag, "priority", "1")
      tag = _set(tag, "friction", "0.6")
    else:
      tag = _set(tag, "condim", "1")
    return f"{tag}{end}"

  return re.sub(r"(<geom\b[^>]*?)(\s*/?>)", _repl, inner)


def _inject_joint_frictionloss(
  inner: str,
  joints: list[dict[str, Any]],
) -> str:
  """Inject frictionloss from deployment_alignment (None → skip)."""
  fl_by_name = {
    str(j["name"]): float(j["frictionloss"])
    for j in joints
    if j.get("controlled", True) and j.get("frictionloss") is not None
  }
  if not fl_by_name:
    return inner
  return _inject_joint_attr(
    inner,
    list(fl_by_name.keys()),
    "frictionloss",
    fl_by_name.get,
    warn_label="frictionloss",
  )


# Field sets the web player knows how to assemble (deployment actor_frame).
_ACTOR_FRAME_FIELDS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)


# Velocity profile (Unitree-H2-*): actor obs from velocity_env_cfg.
_VELOCITY_FIELDS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
)


class _LenientYamlLoader(yaml.SafeLoader):
  """SafeLoader that reads ``!!python/*`` tags as plain data (env.yaml)."""


def _construct_python_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if suffix.startswith("name:"):
    # ``func: !!python/name:pkg.mod.fn ''`` → "pkg.mod.fn"
    return suffix[len("name:") :]
  return loader.construct_scalar(node)


_LenientYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def _velocity_command_limits(env_yaml: Path) -> dict[str, list[float]]:
  """Envelope of twist ranges over the command_vel curriculum (slider limits)."""
  limits: dict[str, list[float]] = {
    "lin_vel_x": [-1.0, 1.0],
    "lin_vel_y": [-0.5, 0.5],
    "ang_vel_z": [-1.0, 1.0],
  }
  if not env_yaml.is_file():
    return limits
  try:
    env = yaml.load(env_yaml.read_text(), Loader=_LenientYamlLoader) or {}
  except yaml.YAMLError as exc:
    print(f"[WARN] cannot parse {env_yaml}: {exc}", file=sys.stderr)
    return limits

  def _merge(ranges: Any) -> None:
    if not isinstance(ranges, dict):
      return
    for key in limits:
      r = ranges.get(key)
      if isinstance(r, (list, tuple)) and len(r) == 2:
        limits[key][0] = min(limits[key][0], float(r[0]))
        limits[key][1] = max(limits[key][1], float(r[1]))

  twist = ((env.get("commands") or {}).get("twist") or {})
  _merge(twist.get("ranges"))
  stages = (
    ((env.get("curriculum") or {}).get("command_vel") or {})
    .get("params", {})
    .get("velocity_stages")
  )
  for stage in stages or []:
    _merge(stage)
  return limits


def _load_alignment(path: Path) -> dict[str, Any]:
  data = yaml.safe_load(path.read_text())
  if not isinstance(data, dict):
    raise ValueError(f"Invalid alignment YAML: {path}")
  return data


def _expand_action_scales(
  joints: list[dict[str, Any]], scale_cfg: float | dict[str, float] | None
) -> list[float]:
  """Expand Isaac/MuJoCo-style regex action_scale map to per-joint floats."""
  names = [j["name"] for j in joints if j.get("controlled", True)]
  if scale_cfg is None:
    return [1.0] * len(names)
  if isinstance(scale_cfg, (int, float)):
    return [float(scale_cfg)] * len(names)
  compiled = [(re.compile(pattern), float(value)) for pattern, value in scale_cfg.items()]
  out: list[float] = []
  for name in names:
    matched = None
    for pattern, value in compiled:
      if pattern.fullmatch(name):
        matched = value
        break
    if matched is None:
      raise ValueError(f"No action_scale pattern matched joint {name}")
    out.append(matched)
  return out


def _onnx_input_dim(onnx_path: Path) -> int | None:
  try:
    import onnx
  except ImportError:
    return None
  model = onnx.load(str(onnx_path))
  if not model.graph.input:
    return None
  dims = model.graph.input[0].type.tensor_type.shape.dim
  values = [d.dim_value for d in dims if d.dim_value > 0]
  if not values:
    return None
  return int(values[-1])


def _find_onnx(run_dir: Path) -> Path | None:
  preferred = run_dir / "policy.onnx"
  if preferred.is_file():
    return preferred
  named = run_dir / f"{run_dir.name}.onnx"
  if named.is_file():
    return named
  candidates = sorted(run_dir.glob("*.onnx"))
  return candidates[0] if candidates else None


def _list_pt_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  """Return ``(step, path)`` for ``model_*.pt``, sorted by step."""
  entries: list[tuple[int, Path]] = []
  for path in run_dir.glob("model_*.pt"):
    try:
      step = int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
      continue
    entries.append((step, path))
  entries.sort(key=lambda item: item[0])
  return entries


def _load_agent_cfg(run_dir: Path) -> dict[str, Any]:
  agent_path = run_dir / "params" / "agent.yaml"
  if not agent_path.is_file():
    raise FileNotFoundError(f"Missing {agent_path}")
  # agent.yaml uses !!python/tuple tags.
  data = yaml.load(agent_path.read_text(), Loader=yaml.UnsafeLoader)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid agent.yaml: {agent_path}")
  return data


def _infer_history_export_dims(
  run_dir: Path, actor_cfg: dict[str, Any]
) -> tuple[int, int, int, int]:
  """Return ``(actor_dim, history_length, history_frame_dim, privileged_dim)`` from ckpt."""
  import torch

  history_length = int(actor_cfg.get("history_length") or 50)
  pts = _list_pt_checkpoints(run_dir)
  if not pts:
    raise FileNotFoundError(
      f"HistoryEncoder export needs model_*.pt under {run_dir} to infer dims"
    )
  payload = torch.load(pts[0][1], map_location="cpu", weights_only=False)
  state = payload.get("actor_state_dict") or {}
  actor_dim = int(state["obs_normalizer._mean"].shape[-1])
  hist_flat = int(state["encoder.0.weight"].shape[1])
  if hist_flat % history_length != 0:
    raise ValueError(
      f"encoder input {hist_flat} not divisible by history_length={history_length}"
    )
  history_frame_dim = hist_flat // history_length
  privileged_dim = int(state["decoder.4.weight"].shape[0])
  return actor_dim, history_length, history_frame_dim, privileged_dim


def _build_actor_for_export(
  agent_cfg: dict[str, Any],
  obs_dim: int,
  act_dim: int,
  *,
  run_dir: Path | None = None,
  actor_dim: int | None = None,
  history_length: int | None = None,
  history_frame_dim: int | None = None,
  privileged_dim: int | None = None,
) -> Any:
  """Build an rsl_rl actor matching training (plain MLP or HistoryEncoder) for ONNX export."""
  import torch
  from tensordict import TensorDict
  from rsl_rl.models.mlp_model import MLPModel

  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

  actor_cfg = dict(agent_cfg.get("actor") or {})
  class_name = str(actor_cfg.get("class_name") or "")
  is_history = "HistoryEncoderMLPModel" in class_name
  for key in (
    "class_name",
    "cnn_cfg",
    "rnn_type",
    "rnn_hidden_dim",
    "rnn_num_layers",
    "num_experts",
    "gate_hidden_dims",
    "top_k",
  ):
    actor_cfg.pop(key, None)
  dist = actor_cfg.get("distribution_cfg")
  if isinstance(dist, dict):
    actor_cfg["distribution_cfg"] = dict(dist)

  if not is_history:
    obs = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
    return MLPModel(obs, {"actor": ["actor"]}, "actor", act_dim, **actor_cfg)

  from src.tasks.tracking.rl.history_actor import HistoryEncoderMLPModel

  if (
    actor_dim is None
    or history_length is None
    or history_frame_dim is None
    or privileged_dim is None
  ):
    if run_dir is None:
      raise ValueError("HistoryEncoder export requires run_dir or explicit dims")
    actor_dim, history_length, history_frame_dim, privileged_dim = (
      _infer_history_export_dims(run_dir, agent_cfg.get("actor") or {})
    )
  assert actor_dim is not None and history_length is not None
  assert history_frame_dim is not None and privileged_dim is not None
  expected = actor_dim + history_length * history_frame_dim
  if obs_dim and obs_dim != expected:
    raise ValueError(
      f"HistoryEncoder obs_dim {obs_dim} != actor({actor_dim})+"
      f"history({history_length}x{history_frame_dim}={history_length * history_frame_dim})"
    )
  obs = TensorDict(
    {
      "actor": torch.zeros(1, actor_dim),
      "actor_history": torch.zeros(1, history_length * history_frame_dim),
      "privileged": torch.zeros(1, privileged_dim),
    },
    batch_size=[1],
  )
  return HistoryEncoderMLPModel(
    obs, {"actor": ["actor"]}, "actor", act_dim, **actor_cfg
  )


def _export_one_pt_to_onnx(
  actor: Any,
  pt_path: Path,
  out_path: Path,
) -> dict[str, Any]:
  """Load ``actor_state_dict`` from ``pt_path`` and write ONNX to ``out_path``.

  Returns the loaded checkpoint payload.
  """
  import torch

  payload = torch.load(pt_path, map_location="cpu", weights_only=False)
  state = payload.get("actor_state_dict")
  if state is None:
    raise ValueError(f"{pt_path.name} missing actor_state_dict")
  actor.load_state_dict(state, strict=True)
  onnx_model = actor.as_onnx(False)
  onnx_model.eval()
  # Write to a temp file then rename — avoids serving a half-written ONNX.
  tmp = out_path.with_suffix(out_path.suffix + ".tmp")
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    str(tmp),
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
  tmp.replace(out_path)
  return payload


_AMP_DISC_SPEC = "amp_disc.json"


def _amp_disc_tensors(payload: dict[str, Any]) -> list[tuple[str, Any, str]]:
  """``(name, float tensor, dtype)`` in a fixed order; normalizer stays fp32."""
  norm = payload.get("amp_normalizer_state_dict")
  disc = payload.get("discriminator_state_dict")
  if norm is None or disc is None:
    raise ValueError("checkpoint missing discriminator / amp_normalizer state")
  out = [("norm.mean", norm["mean"], "f32"), ("norm.var", norm["var"], "f32")]
  out.extend((k, v, "f16") for k, v in disc.items())
  return out


def _write_amp_disc_bin(payload: dict[str, Any], out_path: Path) -> list[dict[str, Any]]:
  """Write discriminator + normalizer as one flat binary; returns tensor layout."""
  import numpy as np

  layout: list[dict[str, Any]] = []
  chunks: list[bytes] = []
  offset = 0
  for name, tensor, dtype in _amp_disc_tensors(payload):
    arr = tensor.detach().cpu().numpy().astype(np.float32 if dtype == "f32" else np.float16)
    raw = arr.tobytes(order="C")
    layout.append(
      {"name": name, "shape": list(arr.shape), "dtype": dtype, "offset": offset}
    )
    chunks.append(raw)
    offset += len(raw)
  tmp = out_path.with_suffix(out_path.suffix + ".tmp")
  tmp.write_bytes(b"".join(chunks))
  tmp.replace(out_path)
  return layout


def _amp_cfg(agent_cfg: dict[str, Any]) -> dict[str, Any]:
  """AMP settings live under ``algorithm`` in agent.yaml."""
  alg = agent_cfg.get("algorithm")
  return alg if isinstance(alg, dict) and alg.get("amp_motion_files") else agent_cfg


def _amp_disc_spec(agent_cfg: dict[str, Any], layout: list[dict[str, Any]]) -> dict[str, Any]:
  from rsl_rl.algorithms.amp_constants import AMP_LABEL_NAMES
  from rsl_rl.modules.amp_ls_discriminator import (
    AmpLSPartGateDiscriminator,
    build_amp_part_indices,
  )

  agent_cfg = _amp_cfg(agent_cfg)
  joint_names = list(agent_cfg["amp_joint_names"])
  key_bodies = list(agent_cfg["amp_key_body_names"])
  disc_type = str(agent_cfg.get("amp_discriminator_type") or "mlp")
  indices = build_amp_part_indices(joint_names, key_bodies)
  parts = [
    {"name": n, "indices": indices[n]}
    for n in AmpLSPartGateDiscriminator.PART_ORDER
    if indices[n]
  ]
  return {
    "type": disc_type,
    "amp_obs_dim": 6 + 2 * len(joint_names) + 9 * len(key_bodies),
    "joint_names": joint_names,
    "key_body_names": key_bodies,
    "label_names": list(AMP_LABEL_NAMES),
    "velocity_threshold": float(agent_cfg.get("amp_motion_velocity_threshold", 0.8)),
    "reward_coefficient": float(agent_cfg.get("amp_reward_coefficient", 0.4)),
    "task_reward_lerp": float(agent_cfg.get("amp_task_reward_lerp", 0.7)),
    "normalizer_epsilon": 1.0e-4,
    "normalizer_clip": 10.0,
    "parts": parts,
    "layout": layout,
  }


def _export_davis_checkpoints(
  run_dir: Path,
  pack_dir: Path,
  *,
  act_dim: int,
  preferred_pt: Path | None = None,
  skip_existing: bool = False,
  max_export: int | None = 3,
) -> tuple[list[dict[str, Any]], str, int]:
  """Export DAVIS dual-input (proprio, depth) ONNX checkpoints.

  DavisActor ONNX is heavier than MLP; by default only the newest few
  checkpoints are exported (plus ``preferred_pt``). Pass ``max_export=None``
  to export every ``model_*.pt``.
  """
  import sys

  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
  scripts_dir = Path(__file__).resolve().parent
  if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

  from davis_web_pack import build_davis_actor_for_export

  ckpt_dir = pack_dir / "ckpts"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  pts = _list_pt_checkpoints(run_dir)
  if not pts:
    raise FileNotFoundError(f"No model_*.pt in {run_dir} (DAVIS needs .pt export)")

  # Prefer latest + preferred; keep list short for CNN export time.
  selected: list[tuple[int, Path]] = []
  if preferred_pt is not None:
    pref = preferred_pt.resolve()
    for step, path in pts:
      if path.resolve() == pref or path.name == preferred_pt.name:
        selected.append((step, path))
        break
  if max_export is None:
    selected = list(pts)
  else:
    tail = pts[-max(1, int(max_export)) :]
    seen = {p.resolve() for _, p in selected}
    for step, path in tail:
      if path.resolve() not in seen:
        selected.append((step, path))
        seen.add(path.resolve())
    selected.sort(key=lambda item: item[0])

  agent_cfg = _load_agent_cfg(run_dir)
  actor = build_davis_actor_for_export(agent_cfg, act_dim=act_dim)
  entries: list[dict[str, Any]] = []
  newly = 0
  for step, pt_path in selected:
    out_name = f"{pt_path.stem}.onnx"
    out_path = ckpt_dir / out_name
    entry: dict[str, Any] = {
      "name": pt_path.name,
      "step": step,
      "file": f"ckpts/{out_name}",
    }
    fresh = (
      skip_existing
      and out_path.is_file()
      and out_path.stat().st_mtime >= pt_path.stat().st_mtime
    )
    if fresh:
      entries.append(entry)
      continue
    try:
      _export_one_pt_to_onnx(actor, pt_path, out_path)
    except Exception as exc:
      if skip_existing:
        print(f"[pack] skip {pt_path.name} ({exc})")
        continue
      raise
    newly += 1
    entries.append(entry)
    print(f"[pack] davis onnx {pt_path.name} → {out_name}")

  if not entries:
    raise FileNotFoundError(f"No exportable DAVIS checkpoints in {run_dir}")

  default_name = entries[-1]["name"]
  if preferred_pt is not None:
    pref_name = preferred_pt.name
    for e in entries:
      if e["name"] == pref_name:
        default_name = pref_name
        break
  default_entry = next(e for e in entries if e["name"] == default_name)
  shutil.copy2(pack_dir / default_entry["file"], pack_dir / "policy.onnx")
  return entries, default_name, newly


def _export_run_checkpoints(
  run_dir: Path,
  pack_dir: Path,
  *,
  obs_dim: int,
  act_dim: int,
  preferred_pt: Path | None = None,
  skip_existing: bool = False,
  actor_dim: int | None = None,
  history_length: int | None = None,
  history_frame_dim: int | None = None,
  privileged_dim: int | None = None,
) -> tuple[list[dict[str, Any]], str, int]:
  """Export ``model_*.pt`` to ``ckpts/model_N.onnx``.

  Returns ``(checkpoint_entries, default_name, newly_exported)``.
  If no ``.pt`` files exist, falls back to the run's ``policy.onnx``.
  When ``skip_existing`` is True, keep ONNX that already exist (for live sync).
  """
  ckpt_dir = pack_dir / "ckpts"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  pts = _list_pt_checkpoints(run_dir)

  if not pts:
    src = _find_onnx(run_dir)
    if src is None:
      raise FileNotFoundError(f"No model_*.pt or .onnx in {run_dir}")
    dest = ckpt_dir / "policy.onnx"
    shutil.copy2(src, dest)
    data_sidecar = Path(str(src) + ".data")
    if data_sidecar.is_file():
      shutil.copy2(data_sidecar, Path(str(dest) + ".data"))
    entry = {"name": "policy.onnx", "step": None, "file": "ckpts/policy.onnx"}
    shutil.copy2(dest, pack_dir / "policy.onnx")
    return [entry], "policy.onnx", 0

  agent_cfg = _load_agent_cfg(run_dir)
  actor = _build_actor_for_export(
    agent_cfg,
    obs_dim,
    act_dim,
    run_dir=run_dir,
    actor_dim=actor_dim,
    history_length=history_length,
    history_frame_dim=history_frame_dim,
    privileged_dim=privileged_dim,
  )
  # AMP runs: also ship the discriminator so the browser can show style reward.
  amp_cfg = _amp_cfg(agent_cfg)
  with_disc = bool(amp_cfg.get("amp_motion_files")) and bool(amp_cfg.get("amp_joint_names"))
  spec_path = pack_dir / _AMP_DISC_SPEC
  entries: list[dict[str, Any]] = []
  newly = 0
  for step, pt_path in pts:
    out_name = f"{pt_path.stem}.onnx"
    out_path = ckpt_dir / out_name
    disc_name = f"{pt_path.stem}.disc.bin"
    disc_path = ckpt_dir / disc_name
    entry: dict[str, Any] = {"name": pt_path.name, "step": step, "file": f"ckpts/{out_name}"}
    if with_disc:
      entry["disc"] = f"ckpts/{disc_name}"
    fresh = (
      skip_existing
      and out_path.is_file()
      # Re-export if the .pt is newer (save finished after a partial export).
      and out_path.stat().st_mtime >= pt_path.stat().st_mtime
      and (not with_disc or (disc_path.is_file() and spec_path.is_file()))
    )
    if fresh:
      entries.append(entry)
      continue
    try:
      payload = _export_one_pt_to_onnx(actor, pt_path, out_path)
      if with_disc:
        try:
          layout = _write_amp_disc_bin(payload, disc_path)
          if not spec_path.is_file():
            spec_path.write_text(json.dumps(_amp_disc_spec(agent_cfg, layout), indent=2))
        except Exception as exc:
          print(f"[pack] no discriminator for {pt_path.name} ({exc})")
          entry.pop("disc", None)
    except Exception as exc:
      # Training may still be writing the .pt; skip and retry next poll.
      if skip_existing:
        print(f"[pack] skip {pt_path.name} ({exc})")
        continue
      raise
    newly += 1
    entries.append(entry)

  if not entries:
    raise FileNotFoundError(f"No exportable checkpoints in {run_dir}")

  # Default: CLI checkpoint if present, else highest step.
  default_name = entries[-1]["name"]
  if preferred_pt is not None:
    pref_name = preferred_pt.name
    for e in entries:
      if e["name"] == pref_name:
        default_name = pref_name
        break

  default_entry = next(e for e in entries if e["name"] == default_name)
  shutil.copy2(pack_dir / default_entry["file"], pack_dir / "policy.onnx")
  return entries, default_name, newly


def sync_new_checkpoints(run_dir: Path, pack_dir: Path) -> int:
  """Export any new ``model_*.pt`` into an existing pack and refresh manifest.

  Returns the number of newly written ONNX files. Safe to call while Vite
  serves ``pack_dir`` (updates files in place; does not recreate the directory).
  """
  pack_dir = pack_dir.resolve()
  run_dir = run_dir.resolve()
  manifest_path = pack_dir / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"Missing pack manifest: {manifest_path}")
  manifest = json.loads(manifest_path.read_text())
  obs_dim = int(manifest["obs_dim"])
  act_dim = int(manifest["num_actions"])
  prev = {e["name"] for e in (manifest.get("checkpoints") or []) if e.get("name")}
  obs_layout = str(manifest.get("obs_layout") or "stacked")
  hist_kwargs: dict[str, Any] = {}
  if obs_layout == "actor_plus_history":
    hist_kwargs = {
      "actor_dim": int(manifest.get("frame_dim") or 0) or None,
      "history_length": int(manifest.get("history_length") or 0) or None,
      "history_frame_dim": int(manifest.get("history_frame_dim") or 0) or None,
    }

  checkpoints, default_ckpt, newly = _export_run_checkpoints(
    run_dir,
    pack_dir,
    obs_dim=obs_dim,
    act_dim=act_dim,
    preferred_pt=None,
    skip_existing=True,
    **hist_kwargs,
  )
  # Always refresh the list (covers first appearance of a step we previously skipped).
  new_names = [e["name"] for e in checkpoints]
  old_names = [e["name"] for e in (manifest.get("checkpoints") or [])]
  if new_names == old_names and newly == 0 and manifest.get("default_checkpoint") == default_ckpt:
    return 0

  manifest["checkpoints"] = checkpoints
  manifest["default_checkpoint"] = default_ckpt
  if (pack_dir / _AMP_DISC_SPEC).is_file():
    manifest.setdefault("files", {})["amp_disc"] = _AMP_DISC_SPEC
  tmp = manifest_path.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(manifest, indent=2))
  tmp.replace(manifest_path)
  added = [n for n in new_names if n not in prev]
  if added:
    print(f"[pack] synced checkpoints: +{', '.join(added)} (default={default_ckpt})")
  return newly


def _task_id_of(alignment: dict[str, Any]) -> str:
  return str(alignment.get("meta", {}).get("task_id") or "")


def resolve_play_profile(task_id: str) -> str:
  """Match scripts/play.py + play_amp.py viewer routing."""
  tid = task_id or ""
  if tid.startswith("AMP-Walk") or "AMP-Walk-Run" in tid:
    return "amp_twist"
  if tid.startswith("Unitree-H2-"):
    return "velocity"
  if tid.startswith("Davis-Shoot") or "Davis-Shoot" in tid:
    return "davis"
  # Kick + e1_mimic tracking (Victory / dance / backflip): same motion obs layout.
  if "Kick" in tid or "Tracking-" in tid or "Tracking" in tid:
    return "kick"
  raise ValueError(
    f"No web play profile for task_id={task_id!r}. "
    "Supported: AMP-Walk-* (amp_twist), Unitree-H2-* (velocity), "
    "*Kick* / *Tracking-* (kick), Davis-Shoot-* (davis). "
    "Use --task to pack one family at a time (Viser does not mix viewers)."
  )


def _robot_family(robot_xml: Path) -> str:
  return "h2" if robot_xml.resolve() == H2_XML.resolve() else "wbot"


def resolve_robot_xml(task_id: str, num_joints: int) -> Path:
  """Pick MJCF like the task robot, not a single hardcoded model."""
  tid = task_id or ""
  if tid.startswith("Unitree-H2-"):
    if num_joints != 29:
      raise ValueError(f"H2 task {task_id!r} expects 29 joints, got {num_joints}")
    xml = H2_XML
  elif tid.startswith("Davis-Shoot") or "Davis-Shoot" in tid or num_joints == 22:
    # DAVIS uses grafted neck; real XML is written at pack time via davis_web_pack.
    xml = ROBOT_XML_DIR / "wbot_20dof.xml"
  elif "12DOF" in tid.upper() or "12Dof" in tid or num_joints == 12:
    xml = ROBOT_XML_DIR / "wbot_12dof.xml"
  elif num_joints == 20 or "20DOF" in tid.upper() or "20Dof" in tid:
    xml = ROBOT_XML_DIR / "wbot_20dof.xml"
  else:
    raise ValueError(
      f"Cannot resolve robot XML for task={task_id!r} joints={num_joints}"
    )
  if not xml.is_file():
    raise FileNotFoundError(xml)
  return xml


def _ensure_shared_meshes(family: str) -> Path:
  shared = SHARED_ROBOTS / family / "meshes"
  shared.parent.mkdir(parents=True, exist_ok=True)
  if shared.exists():
    return shared
  src = _ROBOT_MESH_SRC[family]
  if not src.is_dir():
    raise FileNotFoundError(f"Missing robot meshes: {src}")
  shutil.copytree(src, shared)
  return shared


def _build_scene_xml(alignment: dict[str, Any], robot_xml: Path) -> str:
  family = _robot_family(robot_xml)
  robot_body = robot_xml.read_text()
  inner = re.sub(r"^\s*<mujoco[^>]*>", "", robot_body, count=1, flags=re.I)
  inner = re.sub(r"</mujoco>\s*$", "", inner, count=1, flags=re.I)
  inner = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes/"', inner, count=1)

  joints = [j for j in alignment["joints"] if j.get("controlled", True)]
  joint_names = [j["name"] for j in joints]
  inner = _inject_joint_armature(inner, joint_names, family)
  inner = _inject_joint_frictionloss(inner, joints)
  if family == "h2":
    inner = _apply_h2_collision(inner)

  init = alignment.get("robot_init", {}) or {}
  pos = init.get("pos", [0.0, 0.0, 0.59])
  inner = re.sub(
    r'(<body name="pelvis" pos=")([^"]+)(")',
    rf'\g<1>{float(pos[0])} {float(pos[1])} {float(pos[2])}\g<3>',
    inner,
    count=1,
  )

  floor_assets = """
  <texture type="2d" name="groundplane" builtin="checker" mark="edge"
           rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
           width="512" height="512"/>
  <material name="groundplane" texture="groundplane" texuniform="true"
            texrepeat="5 5" reflectance="0.1"/>
"""
  if re.search(r"<asset>", inner, flags=re.I):
    inner = re.sub(r"(<asset>)", r"\1" + floor_assets, inner, count=1, flags=re.I)
  else:
    inner = f"  <asset>{floor_assets}  </asset>\n" + inner

  floor_body = """
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"
          friction="1 0.005 0.0001"/>
"""
  if not re.search(r"</worldbody>", inner, flags=re.I):
    raise ValueError("Robot MJCF missing </worldbody>")
  inner = re.sub(
    r"</worldbody>",
    floor_body + "  </worldbody>",
    inner,
    count=1,
    flags=re.I,
  )

  # kp/kd/effort from deployment_alignment (train snapshot) — no pack overrides.
  actuator_lines = ["  <actuator>"]
  for j in joints:
    name = j["name"]
    kp = float(j.get("kp", 20.0))
    kd = float(j.get("kd", 1.0))
    limits = j.get("joint_pos_limits", [-3.14, 3.14])
    lo, hi = float(limits[0]), float(limits[1])
    effort = float(j.get("effort_limit", 50.0))
    actuator_lines.append(
      f'    <position name="{name}" joint="{name}" kp="{kp}" kv="{kd}" '
      f'ctrlrange="{lo} {hi}" forcerange="{-effort} {effort}"/>'
    )
  actuator_lines.append("  </actuator>")
  actuators = "\n".join(actuator_lines)

  # implicitfast: match mjlab BuiltinPosition + armature PD numerical stability.
  return f"""<mujoco model="wbot_web_demo">
  <option timestep="{DEFAULT_SIM_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.3 0.3 0.3"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
{inner}
{actuators}
</mujoco>
"""


# Required for kick WASM (projected_gravity is optional — older runs omit it).
_KICK_LAYOUT_FIELDS = (
  "command",
  "motion_anchor_ori_b",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)
_KICK_OPTIONAL_FIELDS = ("projected_gravity",)


def _fields_from_layout(layout: list[dict[str, Any]], terms: dict[str, Any]) -> list[dict[str, Any]]:
  """Convert deployment ``layout`` list into web FieldSpec entries with slices.

  Group-level ``history_length`` is already baked into ``flatten_dim`` in the
  train snapshot; the browser stacks history itself, so FieldSpec.dim is the
  per-frame size (flatten_dim / history_length).
  """
  fields: list[dict[str, Any]] = []
  offset = 0
  for entry in layout:
    name = str(entry.get("name") or "")
    if not name:
      continue
    flat = int(entry.get("flatten_dim") or (entry.get("shape") or [0])[0] or 0)
    hist = int(entry.get("history_length") or 0)
    if hist > 1 and flat % hist == 0:
      dim = flat // hist
    else:
      dim = flat
    term = terms.get(name) if isinstance(terms, dict) else None
    scale = entry.get("scale")
    if scale is None and isinstance(term, dict):
      scale = term.get("scale")
    spec: dict[str, Any] = {
      "name": name,
      "dim": dim,
      "slice": [offset, offset + dim],
      "scale": float(scale) if scale is not None else 1.0,
    }
    params = entry.get("params")
    if isinstance(params, dict) and params:
      spec["params"] = params
    fields.append(spec)
    offset += dim
  return fields


def _normalize_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
  """Return JSON-serializable alignment with per-joint action scales.

  Supports two policy obs layouts:
  - ``stacked`` (AMP / plain MLP): ``obs = stack(H frames of frame_dim)``.
  - ``actor_plus_history`` (HistoryEncoderMLPModel): ``obs = [actor | H×F]``
    where ``actor_history`` is a separate group (proprio subset, time-major).
  """
  joints = [j for j in alignment["joints"] if j.get("controlled", True)]
  scale_cfg = alignment.get("actions", {}).get("joint_pos", {}).get("scale")
  scales = _expand_action_scales(alignment["joints"], scale_cfg)
  obs_raw = alignment.get("observations", {}) or {}
  actor = obs_raw.get("actor", {})
  terms = actor.get("terms") or {}
  layout = actor.get("layout")
  fields: Any = None
  if isinstance(layout, list) and layout:
    # Kick / tracking: layout is a list of term descriptors.
    if isinstance(layout[0], dict) and "name" in layout[0] and "fields" not in layout[0]:
      fields = _fields_from_layout(layout, terms if isinstance(terms, dict) else {})
    else:
      fields = layout[0].get("fields")
  if not fields:
    frame = terms.get("frame", {}) if isinstance(terms, dict) else {}
    fields = frame.get("fields") if isinstance(frame, dict) else None
  if isinstance(fields, dict):
    fields = [{"name": k, **v} for k, v in fields.items()]
  # Already-normalized / synthesized layouts (e.g. DAVIS enrich) keep fields.
  if not fields and isinstance(actor.get("fields"), list):
    fields = actor["fields"]

  hist = actor.get("history_length")
  history_length = int(hist) if hist not in (None, 0) else 1
  # Prefer per-frame dim from fields (history stripped). Train concatenated_dim
  # already includes history — do not treat it as frame_dim.
  frame_dim = 0
  if isinstance(fields, list) and fields:
    frame_dim = sum(int(f.get("dim") or 0) for f in fields if isinstance(f, dict))
  if not frame_dim:
    frame_dim = int(actor.get("frame_dim") or 0)
  concat_dim = int(actor.get("concatenated_dim") or 0)

  # HistoryEncoder: separate ``actor_history`` group → flat [actor | H×F].
  hist_group = obs_raw.get("actor_history")
  obs_layout = "stacked"
  history_frame_dim = frame_dim
  history_term_names: list[str] = []
  if isinstance(hist_group, dict):
    hist_terms = hist_group.get("terms") or {}
    if hist_terms:
      obs_layout = "actor_plus_history"
      hl = hist_group.get("history_length")
      if hl not in (None, 0):
        history_length = int(hl)
      if isinstance(hist_terms, dict):
        history_term_names = [str(k) for k in hist_terms.keys()]
      elif isinstance(hist_terms, list):
        history_term_names = [str(t) for t in hist_terms]
      name_to_dim = {
        str(f.get("name")): int(f.get("dim") or 0)
        for f in (fields or [])
        if isinstance(f, dict) and f.get("name")
      }
      history_frame_dim = sum(name_to_dim.get(n, 0) for n in history_term_names)
      if not history_frame_dim:
        raise ValueError(
          "actor_history terms have no matching actor field dims: "
          f"{history_term_names}"
        )
      concat_dim = frame_dim + history_length * history_frame_dim

  if obs_layout == "stacked":
    if frame_dim and history_length > 1:
      expected_concat = frame_dim * history_length
      if concat_dim and concat_dim != expected_concat:
        # Snapshot baked history into concat; keep browser concat = frame*hist.
        concat_dim = expected_concat
    elif not concat_dim and frame_dim:
      concat_dim = frame_dim * history_length

  out: dict[str, Any] = {
    "meta": alignment.get("meta", {}),
    "joints": joints,
    "robot_init": alignment.get("robot_init", {}),
    "observations": {
      "obs_layout": obs_layout,
      "actor": {
        # Actor group itself is single-frame for HistoryEncoder; H lives under
        # actor_history / manifest.history_length.
        "history_length": 1 if obs_layout == "actor_plus_history" else history_length,
        "concatenated_dim": concat_dim
        or (
          frame_dim + history_length * history_frame_dim
          if obs_layout == "actor_plus_history"
          else frame_dim * history_length
        ),
        "frame_dim": frame_dim,
        "fields": fields or [],
        "obs_scales": actor.get("obs_scales", {}),
      },
    },
    "actions": {
      "joint_pos": {
        "scale_per_joint": {j["name"]: scales[i] for i, j in enumerate(joints)},
        "scale_list": scales,
      }
    },
    "control": {
      "sim_dt": DEFAULT_SIM_DT,
      "decimation": DEFAULT_DECIMATION,
      "policy_dt": DEFAULT_SIM_DT * DEFAULT_DECIMATION,
    },
    "bodies": {
      "torso": "torso_link",
      "root": "pelvis",
    },
  }
  if obs_layout == "actor_plus_history":
    out["observations"]["actor_history"] = {
      "history_length": history_length,
      "frame_dim": history_frame_dim,
      "terms": history_term_names,
    }
  return out


def _field_names(fields: Any) -> list[str]:
  if isinstance(fields, list):
    return [str(f.get("name")) for f in fields if isinstance(f, dict) and f.get("name")]
  if isinstance(fields, dict):
    return [str(k) for k in fields]
  return []


def _assert_profile(alignment: dict[str, Any], profile: str) -> None:
  """Validate obs layout for the chosen play profile (no cross-task packing)."""
  actor = alignment["observations"]["actor"]
  names = _field_names(actor.get("fields"))
  joints = alignment["joints"]
  n = len(joints)

  if profile == "amp_twist":
    missing = [f for f in _ACTOR_FRAME_FIELDS if f not in names]
    if missing:
      raise ValueError(
        f"amp_twist requires actor_frame fields {list(_ACTOR_FRAME_FIELDS)}; "
        f"missing {missing}; got {names or 'no fields'}"
      )
    if "command" not in names:
      raise ValueError("amp_twist requires command field (Viser UniformVelocityCommand)")
    expected_frame = 9 + 3 * n
    frame_dim = int(actor.get("frame_dim") or 0)
    if frame_dim and frame_dim != expected_frame:
      raise ValueError(
        f"frame_dim {frame_dim} != expected actor_frame {expected_frame}"
      )
    return

  if profile == "kick":
    missing = [f for f in _KICK_LAYOUT_FIELDS if f not in names]
    if missing:
      raise ValueError(
        f"kick requires tracking fields {list(_KICK_LAYOUT_FIELDS)}; "
        f"missing {missing}; got {names or 'no fields'}"
      )
    unknown = [
      f
      for f in names
      if f not in _KICK_LAYOUT_FIELDS and f not in _KICK_OPTIONAL_FIELDS
    ]
    if unknown:
      raise ValueError(
        f"kick has unsupported actor fields {unknown}; "
        f"supported={list(_KICK_LAYOUT_FIELDS) + list(_KICK_OPTIONAL_FIELDS)}"
      )
    # command = joint_pos(n) + joint_vel(n)
    cmd = next(
      (f for f in actor["fields"] if isinstance(f, dict) and f.get("name") == "command"),
      None,
    )
    if cmd and int(cmd.get("dim") or 0) != 2 * n:
      raise ValueError(
        f"kick command dim {cmd.get('dim')} != 2*num_joints={2 * n}"
      )
    return

  if profile == "davis":
    required = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
    missing = [f for f in required if f not in names]
    if missing:
      raise ValueError(
        f"davis requires proprio fields {list(required)}; missing {missing}; got {names}"
      )
    if n != 22:
      raise ValueError(f"davis expects 22 joints (20 body + 2 neck), got {n}")
    frame_dim = int(actor.get("frame_dim") or 0)
    if frame_dim and frame_dim != 72:
      raise ValueError(f"davis frame_dim {frame_dim} != 72")
    return

  if profile == "velocity":
    unknown = [f for f in names if f not in _VELOCITY_FIELDS]
    if unknown:
      raise ValueError(
        f"velocity has unsupported actor fields {unknown}; "
        f"supported={list(_VELOCITY_FIELDS)}"
      )
    if "command" not in names:
      raise ValueError("velocity requires the twist command field")
    expected = {"command": 3, "phase": 2, "joint_pos": n, "joint_vel": n, "actions": n}
    for f in actor["fields"]:
      want = expected.get(f.get("name"))
      if want is not None and int(f.get("dim") or 0) != want:
        raise ValueError(f"velocity field {f['name']} dim {f.get('dim')} != {want}")
    return

  raise ValueError(f"Unknown play profile: {profile}")


def _parse_env_rewards(env_yaml: Path) -> list[dict[str, Any]] | None:
  """Best-effort extract of reward terms from params/env.yaml (no UnsafeLoader)."""
  if not env_yaml.is_file():
    return None
  text = env_yaml.read_text()
  m = re.search(r"\nrewards:\n", text)
  if not m:
    return None
  # Slice until next top-level key (2-space indent section end → `\n[a-z_]+:\n` at col 0).
  rest = text[m.end() :]
  end = re.search(r"\n[a-zA-Z_][a-zA-Z0-9_]*:\n", rest)
  chunk = rest[: end.start()] if end else rest
  terms: list[dict[str, Any]] = []
  # Term blocks: `  name:\n` then indented fields.
  for tm in re.finditer(
    r"^  ([a-zA-Z0-9_]+):\n((?:    .*\n)+)", chunk, flags=re.MULTILINE
  ):
    name = tm.group(1)
    body = tm.group(2)
    wm = re.search(r"^\s+weight:\s*([-+0-9.eE]+)\s*$", body, flags=re.MULTILINE)
    if not wm:
      continue
    weight = float(wm.group(1))
    if weight == 0.0:
      continue
    # Skip contact-sensor terms (no sensor in WASM pack).
    if "self_collision" in body or name == "self_collisions":
      continue
    term: dict[str, Any] = {"name": name, "weight": weight, "kind": "unknown"}
    if "motion_global_anchor_position" in body or name.endswith("root_pos"):
      term["kind"] = "anchor_pos"
    elif "motion_global_anchor_orientation" in body or name.endswith("root_ori"):
      term["kind"] = "anchor_ori"
    elif "motion_relative_body_position" in body or (
      "body_pos" in name and "ee" not in name and "lin_vel" not in name
    ):
      term["kind"] = "body_pos"
    elif "motion_relative_body_orientation" in body or (
      "body_ori" in name and "ee" not in name
    ):
      term["kind"] = "body_ori"
    elif "motion_global_body_linear_velocity" in body or "lin_vel" in name:
      term["kind"] = "body_lin_vel"
    elif "motion_global_body_angular_velocity" in body or "ang_vel" in name:
      term["kind"] = "body_ang_vel"
    elif "action_rate" in name:
      term["kind"] = "action_rate"
    elif "action_smooth" in name or "action_acc" in name:
      term["kind"] = "action_smooth"
    elif "joint_pos_limits" in body or name == "joint_limit":
      term["kind"] = "joint_limit"
    elif (
      "motion_foot_pos_z" in name
      or "foot_pos_z" in name
      or "motion_foot_pos_z_error" in body
    ):
      term["kind"] = "foot_pos_z"
    # EE variants share relative/global funcs but restrict body_names.
    if name.startswith("motion_global_ee_"):
      if "pos" in name and "lin" not in name:
        term["kind"] = "body_pos"
      elif "ori" in name:
        term["kind"] = "body_ori"
      elif "lin_vel" in name:
        term["kind"] = "body_lin_vel"
      elif "ang_vel" in name:
        term["kind"] = "body_ang_vel"

    # std: scalar or map
    sm = re.search(
      r"^\s+std:\s*([-+0-9.eE]+)\s*$", body, flags=re.MULTILINE
    )
    if sm:
      term["std"] = float(sm.group(1))
    else:
      std_map: dict[str, float] = {}
      in_std = False
      for line in body.splitlines():
        if re.match(r"^\s+std:\s*$", line):
          in_std = True
          continue
        if in_std:
          mm = re.match(r"^\s{6,}([a-zA-Z0-9_]+):\s*([-+0-9.eE]+)\s*$", line)
          if mm:
            std_map[mm.group(1)] = float(mm.group(2))
          elif re.match(r"^\s{4}[a-zA-Z_]", line):
            break
      if std_map:
        term["std"] = std_map

    # body_names: !!python/tuple or list
    bn = re.search(
      r"body_names:\s*!!python/tuple\s*\n((?:\s+-\s+.+\n)+)", body
    )
    if not bn:
      bn = re.search(r"body_names:\s*\n((?:\s+-\s+.+\n)+)", body)
    if bn:
      names = re.findall(r"-\s+(.+)", bn.group(1))
      term["body_names"] = [n.strip().strip("'\"") for n in names]

    if term["kind"] == "unknown":
      continue
    # Relative/global body terms without explicit body_names → all tracked bodies.
    if term["kind"].startswith("body_") and not term.get("body_names"):
      std = term.get("std")
      if isinstance(std, dict) and std:
        term["body_names"] = list(std.keys())
      else:
        term["body_names"] = list(_default_kick_body_names())
    if term["kind"] == "foot_pos_z" and not term.get("body_names"):
      term["body_names"] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    terms.append(term)
  return terms or None


def _default_kick_body_names() -> tuple[str, ...]:
  return (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
  )


def _default_kick_rewards() -> list[dict[str, Any]]:
  """Fallback matching ``wbot_20dof`` kick3 env_cfgs (sans self_collisions)."""
  bodies = list(_default_kick_body_names())
  legs = {
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  }

  def std_map(leg: float, other: float) -> dict[str, float]:
    return {n: (leg if n in legs else other) for n in bodies}

  return [
    {"name": "motion_global_root_pos", "kind": "anchor_pos", "weight": 0.5, "std": 0.3},
    {"name": "motion_global_root_ori", "kind": "anchor_ori", "weight": 0.5, "std": 0.4},
    {
      "name": "motion_body_pos",
      "kind": "body_pos",
      "weight": 1.0,
      "std": std_map(0.28, 0.3),
      "body_names": bodies,
    },
    {
      "name": "motion_body_ori",
      "kind": "body_ori",
      "weight": 1.0,
      "std": std_map(0.38, 0.4),
      "body_names": bodies,
    },
    {
      "name": "motion_body_lin_vel",
      "kind": "body_lin_vel",
      "weight": 1.0,
      "std": std_map(0.95, 1.0),
      "body_names": bodies,
    },
    {
      "name": "motion_body_ang_vel",
      "kind": "body_ang_vel",
      "weight": 1.0,
      "std": std_map(3.0, 3.14),
      "body_names": bodies,
    },
    {
      "name": "motion_foot_pos_z",
      "kind": "foot_pos_z",
      "weight": 3.0,
      "std": 0.1,
      "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
    },
    {"name": "action_rate_l2", "kind": "action_rate", "weight": -0.1},
    {"name": "joint_limit", "kind": "joint_limit", "weight": -10.0},
  ]


_VELOCITY_REWARD_KINDS = {
  "track_linear_velocity": "track_lin_vel",
  "track_angular_velocity": "track_ang_vel",
  "body_orientation_l2": "body_orientation",
  "variable_posture": "variable_posture",
  "body_angular_velocity_penalty": "body_ang_vel_xy",
  "angular_momentum_penalty": "angular_momentum",
  "joint_acc_l2": "joint_acc",
  "joint_pos_limits": "joint_pos_limits",
  "action_rate_l2": "action_rate",
  "feet_gait": "feet_gait",
  "feet_clearance": "feet_clearance",
  "feet_slip": "feet_slip",
  "soft_landing": "soft_landing",
  "stand_still": "stand_still",
  "self_collision_cost": "self_collision",
}
# Always 0 in browser play (no terminations).
_VELOCITY_REWARD_SKIP = {"is_terminated"}


def _resolve_joint_values(
  pattern_map: dict[str, Any], joint_names: list[str]
) -> dict[str, float]:
  out: dict[str, float] = {}
  for jn in joint_names:
    for pat, val in pattern_map.items():
      if re.fullmatch(pat, jn):
        out[jn] = float(val)
        break
  return out


def _tuple_names(v: Any) -> list[str]:
  if isinstance(v, str):
    return [v]
  if isinstance(v, (list, tuple)):
    return [str(x) for x in v]
  return []


def _export_velocity_rewards_json(
  run_dir: Path,
  out_path: Path,
  joint_names: list[str],
  root_body: str,
) -> dict[str, Any] | None:
  """Velocity task reward specs for live evaluation in the browser."""
  env_yaml = run_dir / "params" / "env.yaml"
  if not env_yaml.is_file():
    return None
  try:
    env = yaml.load(env_yaml.read_text(), Loader=_LenientYamlLoader) or {}
  except yaml.YAMLError as exc:
    print(f"[WARN] cannot parse {env_yaml}: {exc}", file=sys.stderr)
    return None
  rewards = env.get("rewards") or {}
  if not isinstance(rewards, dict) or not rewards:
    return None
  scene = env.get("scene") or {}
  sensors = {
    s.get("name"): s for s in (scene.get("sensors") or []) if isinstance(s, dict)
  }
  robot = ((scene.get("entities") or {}).get("robot") or {})
  soft_factor = float(
    (robot.get("articulation") or {}).get("soft_joint_pos_limit_factor", 1.0)
  )

  terms: list[dict[str, Any]] = []
  unsupported: list[str] = []
  for name, cfg in rewards.items():
    if not isinstance(cfg, dict):
      continue
    weight = float(cfg.get("weight") or 0.0)
    if weight == 0.0:
      continue
    func = str(cfg.get("func") or "").rsplit(".", 1)[-1]
    if func in _VELOCITY_REWARD_SKIP:
      continue
    params = cfg.get("params") or {}
    asset_cfg = params.get("asset_cfg") or {}
    kind = _VELOCITY_REWARD_KINDS.get(func)
    term: dict[str, Any] = {"name": name, "kind": kind or "unsupported", "weight": weight}
    if kind is None:
      unsupported.append(f"{name}({func})")
      terms.append(term)
      continue
    for key in ("std", "command_threshold", "target_height", "period", "threshold",
                "force_threshold", "walking_threshold", "running_threshold"):
      if key in params and isinstance(params[key], (int, float)):
        term[key] = float(params[key])
    if "offset" in params:
      term["offset"] = [float(x) for x in params["offset"]]
    bodies = _tuple_names(asset_cfg.get("body_names"))
    if kind in ("body_orientation", "body_ang_vel_xy"):
      term["body"] = bodies[0] if bodies else root_body
    elif kind == "angular_momentum":
      term["body"] = root_body
    sites = _tuple_names(asset_cfg.get("site_names"))
    if sites:
      term["site_names"] = sites
    if kind == "variable_posture":
      for key in ("std_standing", "std_walking", "std_running"):
        term[key] = _resolve_joint_values(params.get(key) or {}, joint_names)
    elif kind == "joint_pos_limits":
      term["soft_factor"] = soft_factor
    sensor = sensors.get(params.get("sensor_name"))
    if sensor is not None:
      primary = sensor.get("primary") or {}
      term["contact"] = {
        "pattern": str(primary.get("pattern") or ""),
        "mode": str(primary.get("mode") or "body"),
        "history_length": int(sensor.get("history_length") or 0),
      }
    terms.append(term)

  if unsupported:
    print(f"[pack] velocity rewards not evaluated in browser: {', '.join(unsupported)}")
  payload = {"profile": "velocity", "root_body": root_body, "terms": terms}
  out_path.write_text(json.dumps(payload, indent=2))
  return {"n_terms": len(terms), "source": "env.yaml"}


_AMP_REWARD_KINDS = {
  "track_linear_velocity": "track_lin_vel_xy",
  "track_yaw_velocity": "track_yaw_vel",
  "yaw_rate_error_l2": "yaw_rate_error",
  "lin_vel_xy_error_l2": "lin_vel_xy_error",
  "straight_walk_reward": "straight_walk",
  "straight_walk_penalty": "straight_walk_penalty",
  "heading_hold": "heading_hold",
  "zero_command_joint_l2": "zero_command_joint",
  "is_alive": "alive",
  "vertical_velocity_l2": "lin_vel_z",
  "base_angular_velocity_xy_l2": "root_ang_vel_xy",
  "joint_vel_l2": "joint_vel",
  "joint_acc_l2": "joint_acc",
  "action_rate_l2": "action_rate",
  "action_acc_l2": "action_acc",
  "joint_pos_limits": "joint_pos_limits",
  "energy_l2": "energy",
  "flat_orientation_l2": "body_orientation",
  "torso_orientation_l2": "body_orientation",
  "torso_ang_vel_xy_l2": "body_ang_vel_xy",
  "torso_roll_l2": "body_roll",
  "torso_roll_ang_vel_l2": "body_roll_ang_vel",
  "stand_torso_pitch_l2": "stand_body_pitch",
  "stand_feet_heading_l2": "stand_feet_heading",
  "feet_pelvis_heading_l2": "feet_pelvis_heading",
  "stable_stand_hip_roll_norm": "stable_stand_joint_norm",
  "stable_stand_feet_fore_aft_l1": "stable_stand_feet_fore_aft",
  "stable_stand_feet_lateral_center_l1": "stable_stand_feet_lateral",
  "soft_landing": "soft_landing",
  "feet_slide": "feet_slide",
  "walk_swing_foot_clearance": "walk_swing_clearance",
  "walk_air_time_tracking": "walk_air_time",
  "touchdown_foot_velocity": "touchdown_foot_vel",
  "turn_clearance_lift": "turn_clearance_lift",
  "turn_foot_height_tracking": "turn_foot_height",
  "turn_lateral_air_time": "turn_lateral_air_time",
  "turn_lateral_contact_pattern": "turn_lateral_contact",
  "turn_lateral_air_time_dense": "turn_lateral_air_dense",
  "feet_crossing": "feet_crossing",
  "feet_spacing_non_lateral": "feet_spacing",
  "feet_support_roll_l2": "feet_support_roll",
  "ankle_roll_action_rate_l2": "joint_action_rate_loaded",
  "ankle_roll_l2": "joint_dev_norm",
  "undesired_contacts": "undesired_contacts",
  "self_collision_cost": "self_collision",
}
_AMP_TURN_PROFILE_DIR = ROOT / "src" / "assets" / "amp_motions" / "wbot_20dof" / "walk_run_amp"


def _amp_turn_profiles() -> dict[str, Any] | None:
  """Expert turn foot-height curves for ``turn_foot_height_tracking``."""
  import numpy as np

  out: dict[str, Any] = {}
  for key, fname in (("left_turn", "turn_l"), ("right_turn", "turn_r")):
    path = _AMP_TURN_PROFILE_DIR / f"{fname}_foot_height_profile.npz"
    if not path.is_file():
      return None
    data = np.load(path)
    out[key] = {
      "left": [float(x) for x in data["left"]],
      "right": [float(x) for x in data["right"]],
      "period_s": float(data["period_s"]),
    }
  return out


def _export_amp_rewards_json(
  run_dir: Path,
  out_path: Path,
  joint_names: list[str],
  root_body: str,
) -> dict[str, Any] | None:
  """AMP walk/run task reward specs (``src.tasks.amp.mdp.rewards``) for the browser."""
  env_yaml = run_dir / "params" / "env.yaml"
  if not env_yaml.is_file():
    return None
  try:
    env = yaml.load(env_yaml.read_text(), Loader=_LenientYamlLoader) or {}
  except yaml.YAMLError as exc:
    print(f"[WARN] cannot parse {env_yaml}: {exc}", file=sys.stderr)
    return None
  rewards = env.get("rewards") or {}
  if not isinstance(rewards, dict) or not rewards:
    return None
  scene = env.get("scene") or {}
  sensors = {
    s.get("name"): s for s in (scene.get("sensors") or []) if isinstance(s, dict)
  }
  robot = ((scene.get("entities") or {}).get("robot") or {})
  soft_factor = float(
    (robot.get("articulation") or {}).get("soft_joint_pos_limit_factor", 1.0)
  )

  def _joints(cfg: Any) -> list[str] | None:
    pats = _tuple_names((cfg or {}).get("joint_names"))
    if not pats:
      return None
    return [jn for jn in joint_names if any(re.fullmatch(p, jn) for p in pats)]

  terms: list[dict[str, Any]] = []
  unsupported: list[str] = []
  turn_profiles: dict[str, Any] | None = None
  for name, cfg in rewards.items():
    if not isinstance(cfg, dict):
      continue
    weight = float(cfg.get("weight") or 0.0)
    if weight == 0.0:
      continue
    func = str(cfg.get("func") or "").rsplit(".", 1)[-1]
    if func in _VELOCITY_REWARD_SKIP:
      continue
    kind = _AMP_REWARD_KINDS.get(func)
    term: dict[str, Any] = {"name": name, "kind": kind or "unsupported", "weight": weight}
    terms.append(term)
    if kind is None:
      unsupported.append(f"{name}({func})")
      continue
    params = cfg.get("params") or {}
    term["params"] = {
      k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
      for k, v in params.items()
      if isinstance(v, (bool, int, float))
    }
    joints = _joints(params.get("asset_cfg"))
    if joints is not None:
      term["joints"] = joints
    body_cfg = params.get("torso_cfg") or params.get("asset_cfg") or {}
    bodies = _tuple_names(body_cfg.get("body_names"))
    term["body"] = bodies[0] if bodies else root_body
    feet = _tuple_names((params.get("feet_cfg") or {}).get("body_names"))
    if feet:
      term["feet"] = feet
    if kind == "joint_pos_limits":
      term["soft_factor"] = soft_factor
    if kind == "turn_foot_height":
      turn_profiles = turn_profiles or _amp_turn_profiles()
      if turn_profiles is None:
        term["kind"] = "unsupported"
        unsupported.append(f"{name}(missing turn profile npz)")
    sensor = sensors.get(params.get("sensor_name"))
    if sensor is not None:
      primary = sensor.get("primary") or {}
      term["contact"] = {
        "pattern": str(primary.get("pattern") or ""),
        "mode": str(primary.get("mode") or "body"),
        "history_length": int(sensor.get("history_length") or 0),
      }

  if unsupported:
    print(f"[pack] amp rewards not evaluated in browser: {', '.join(unsupported)}")
  payload: dict[str, Any] = {"profile": "amp", "root_body": root_body, "terms": terms}
  if turn_profiles is not None:
    payload["turn_profiles"] = turn_profiles
  out_path.write_text(json.dumps(payload, indent=2))
  return {"n_terms": len(terms), "source": "env.yaml"}


def _export_rewards_json(run_dir: Path, out_path: Path, profile: str) -> dict[str, Any] | None:
  """Write browser reward term specs (Viser Rewards bar parity for kick)."""
  if profile != "kick":
    return None
  terms = _parse_env_rewards(run_dir / "params" / "env.yaml") or _default_kick_rewards()
  payload = {
    "anchor_body": "torso_link",
    "terms": terms,
  }
  out_path.write_text(json.dumps(payload, indent=2))
  return {"n_terms": len(terms), "source": "env.yaml" if terms else "default"}


def _export_kick_motion_json(motion_path: Path, out_path: Path) -> dict[str, Any]:
  """Write browser motion JSON (joints + root/anchor pose for ghost viz)."""
  import numpy as np

  data = np.load(motion_path)
  body_names = [str(x) for x in data["body_names"]]
  joint_names = [str(x) for x in data["joint_names"]]
  try:
    anchor_idx = body_names.index("torso_link")
  except ValueError as exc:
    raise ValueError(f"motion missing torso_link in body_names: {body_names}") from exc
  try:
    root_idx = body_names.index("pelvis")
  except ValueError:
    root_idx = 0
  fps = int(np.asarray(data["fps"]).reshape(-1)[0])
  joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
  joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
  body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
  body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
  payload = {
    "fps": fps,
    "T": int(joint_pos.shape[0]),
    "joint_names": joint_names,
    "anchor_body": "torso_link",
    "anchor_body_index": anchor_idx,
    "root_body": body_names[root_idx],
    "root_body_index": root_idx,
    "joint_pos": joint_pos.reshape(-1).tolist(),
    "joint_vel": joint_vel.reshape(-1).tolist(),
    "anchor_pos_w": body_pos[:, anchor_idx, :].reshape(-1).tolist(),
    "anchor_quat_w": body_quat[:, anchor_idx, :].reshape(-1).tolist(),
    "root_pos_w": body_pos[:, root_idx, :].reshape(-1).tolist(),
    "root_quat_w": body_quat[:, root_idx, :].reshape(-1).tolist(),
  }
  out_path.write_text(json.dumps(payload))
  return {
    "fps": fps,
    "T": payload["T"],
    "anchor_body": "torso_link",
    "root_body": body_names[root_idx],
  }

def _prepare_pack_dir(pack_dir: Path) -> None:
  """Empty ``pack_dir`` while keeping the directory itself.

  Vite's public-file server caches paths; ``shutil.rmtree`` + recreate while
  ``npm run dev`` is running leaves nested URLs as sticky 404s until restart.
  """
  if pack_dir.is_dir():
    for child in pack_dir.iterdir():
      if child.is_dir() and not child.is_symlink():
        shutil.rmtree(child)
      else:
        child.unlink(missing_ok=True)
    return
  if pack_dir.exists():
    pack_dir.unlink()
  pack_dir.mkdir(parents=True)


def pack_run(
  run_dir: Path,
  out_root: Path,
  *,
  motion_file: Path | str | None = None,
  checkpoint_file: Path | str | None = None,
) -> dict[str, Any]:
  run_dir = run_dir.resolve()
  alignment_path = run_dir / "params" / "deployment_alignment.yaml"
  if not alignment_path.is_file():
    raise FileNotFoundError(f"Missing {alignment_path}")
  if not _list_pt_checkpoints(run_dir) and _find_onnx(run_dir) is None:
    raise FileNotFoundError(f"No model_*.pt or .onnx in {run_dir}")

  alignment_raw = _load_alignment(alignment_path)
  task_id = _task_id_of(alignment_raw)
  profile = resolve_play_profile(task_id)
  if profile == "davis":
    from davis_web_pack import enrich_davis_alignment

    enrich_davis_alignment(alignment_raw)
  if "actor" not in alignment_raw.get("observations", {}):
    raise ValueError("alignment missing observations.actor")

  # Optional CLI override (play.py --motion-file).
  if motion_file is not None:
    mf = Path(motion_file).expanduser().resolve()
    if not mf.is_file():
      raise FileNotFoundError(f"motion_file not found: {mf}")
    alignment_raw.setdefault("meta", {})["motion_file"] = str(mf)

  alignment = _normalize_alignment(alignment_raw)
  alignment["meta"] = dict(alignment.get("meta") or {})
  alignment["meta"]["play_profile"] = profile
  if profile == "davis":
    # normalize rebuilds actor from layout/terms; re-apply DAVIS fields + depth.
    from davis_web_pack import enrich_davis_alignment

    enrich_davis_alignment(alignment)
  _assert_profile(alignment, profile)

  joints = alignment["joints"]
  obs_layout = str(alignment["observations"].get("obs_layout") or "stacked")
  frame_dim = int(alignment["observations"]["actor"]["frame_dim"] or 0)
  history_frame_dim = frame_dim
  history_term_names: list[str] = []
  if obs_layout == "actor_plus_history":
    ah = alignment["observations"].get("actor_history") or {}
    hist = int(ah.get("history_length") or 1)
    history_frame_dim = int(ah.get("frame_dim") or 0)
    history_term_names = [str(t) for t in (ah.get("terms") or [])]
    obs_dim = int(alignment["observations"]["actor"]["concatenated_dim"] or 0)
    if not obs_dim:
      obs_dim = frame_dim + hist * history_frame_dim
  else:
    hist = int(alignment["observations"]["actor"]["history_length"] or 1)
    obs_dim = int(alignment["observations"]["actor"]["concatenated_dim"] or 0)
    if not obs_dim:
      obs_dim = frame_dim * hist
  act_dim = len(joints)

  # Prefer obs_dim from an existing ONNX if alignment concat is missing.
  # DAVIS ONNX is dual-input (proprio + depth) — skip single-input dim check.
  if profile != "davis":
    existing_onnx = _find_onnx(run_dir)
    if existing_onnx is not None:
      onnx_dim = _onnx_input_dim(existing_onnx)
      if onnx_dim is not None:
        if obs_dim and onnx_dim != obs_dim:
          raise ValueError(
            f"ONNX input dim {onnx_dim} != alignment concatenated_dim {obs_dim}"
          )
        obs_dim = obs_dim or onnx_dim

  robot_xml = resolve_robot_xml(task_id, act_dim)

  pack_id = run_dir.name
  pack_dir = out_root / pack_id
  # Clear in place — do NOT rmtree+recreate. Vite caches public/ paths; removing
  # the directory while `npm run dev` is up makes nested files 404 until restart.
  _prepare_pack_dir(pack_dir)

  preferred_pt = Path(checkpoint_file) if checkpoint_file else None
  print(f"[pack] exporting checkpoints from {run_dir.name}…")
  hist_export: dict[str, Any] = {}
  if obs_layout == "actor_plus_history":
    hist_export = {
      "actor_dim": frame_dim,
      "history_length": hist,
      "history_frame_dim": history_frame_dim,
    }
  if profile == "davis":
    checkpoints, default_ckpt, _newly = _export_davis_checkpoints(
      run_dir, pack_dir, act_dim=act_dim, preferred_pt=preferred_pt
    )
  else:
    checkpoints, default_ckpt, _newly = _export_run_checkpoints(
      run_dir,
      pack_dir,
      obs_dim=obs_dim,
      act_dim=act_dim,
      preferred_pt=preferred_pt,
      **hist_export,
    )
  print(f"[pack] {len(checkpoints)} checkpoint(s); default={default_ckpt}")

  shared_meshes = _ensure_shared_meshes(_robot_family(robot_xml))
  shutil.copytree(shared_meshes, pack_dir / "meshes")

  if profile == "davis":
    from davis_web_pack import build_davis_scene_xml, write_davis_robot_xml

    davis_robot = write_davis_robot_xml(pack_dir)
    (pack_dir / "scene.xml").write_text(build_davis_scene_xml(alignment, davis_robot))
  else:
    (pack_dir / "scene.xml").write_text(_build_scene_xml(alignment_raw, robot_xml))
  (pack_dir / "deployment_alignment.json").write_text(
    json.dumps(alignment, indent=2)
  )
  field_names = _field_names(alignment["observations"]["actor"].get("fields"))
  files = {
    "policy": "policy.onnx",
    "scene": "scene.xml",
    "alignment": "deployment_alignment.json",
  }
  motion_meta: dict[str, Any] | None = None
  rewards_meta: dict[str, Any] | None = None
  if profile == "kick":
    motion_src = Path(str(alignment_raw.get("meta", {}).get("motion_file") or ""))
    if not motion_src.is_file():
      # deployment_alignment often stores repo-relative paths.
      alt = (ROOT / motion_src).resolve()
      if alt.is_file():
        motion_src = alt
    if not motion_src.is_file():
      # Prefer npz copied into the run dir (train/play convention).
      run_npzs = sorted(run_dir.glob("*.npz"))
      if len(run_npzs) == 1:
        motion_src = run_npzs[0]
      elif run_npzs:
        # Prefer names matching task clip over misc dumps.
        preferred = [
          p
          for p in run_npzs
          if "tracking" in p.name.lower() or "victory" in p.name.lower()
          or "kick" in p.name.lower() or "dance" in p.name.lower()
          or "backflip" in p.name.lower()
        ]
        motion_src = preferred[0] if preferred else run_npzs[0]
    if not motion_src.is_file():
      raise FileNotFoundError(
        f"Kick/tracking pack needs meta.motion_file (npz); missing {motion_src}"
      )
    motion_meta = _export_kick_motion_json(motion_src, pack_dir / "motion.json")
    files["motion"] = "motion.json"
    rewards_meta = _export_rewards_json(run_dir, pack_dir / "rewards.json", profile)
    if rewards_meta is not None:
      files["rewards"] = "rewards.json"
  elif profile in ("velocity", "amp_twist"):
    export = (
      _export_velocity_rewards_json if profile == "velocity" else _export_amp_rewards_json
    )
    rewards_meta = export(
      run_dir,
      pack_dir / "rewards.json",
      [j["name"] for j in joints],
      str(alignment["bodies"]["root"]),
    )
    if rewards_meta is not None:
      files["rewards"] = "rewards.json"
  if (pack_dir / _AMP_DISC_SPEC).is_file():
    files["amp_disc"] = _AMP_DISC_SPEC

  command_limits = (
    _velocity_command_limits(run_dir / "params" / "env.yaml")
    if profile == "velocity"
    else None
  )

  if obs_layout == "actor_plus_history":
    packed_obs_dim = obs_dim or (frame_dim + hist * history_frame_dim)
  else:
    packed_obs_dim = obs_dim or frame_dim * hist
  manifest = {
    "id": pack_id,
    "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
    "task_id": task_id,
    "play_profile": profile,
    "robot_xml": robot_xml.name,
    "obs_layout": obs_layout,
    "obs_dim": packed_obs_dim,
    "frame_dim": frame_dim,
    "history_length": hist,
    "history_frame_dim": history_frame_dim if obs_layout == "actor_plus_history" else frame_dim,
    "history_terms": history_term_names,
    "num_actions": act_dim,
    "joint_names": [j["name"] for j in joints],
    "default_joint_pos": [float(j.get("default_joint_pos", 0.0)) for j in joints],
    "action_scale": alignment["actions"]["joint_pos"]["scale_list"],
    "field_names": field_names,
    # AMP twist / velocity use 3-D velocity command; kick "command" is motion joints.
    "has_velocity_command": profile in ("amp_twist", "velocity")
    and "command" in field_names,
    # Tracking clips (getup, dances, ...) reset to the motion frame like mjlab play
    # sampling_mode="start"; training never starts from the standing default pose.
    # Kick keeps the default pose (its clip already starts standing).
    "reset_from_motion": profile == "kick" and "kick" not in task_id.lower(),
    "policy_dt": alignment["control"]["policy_dt"],
    "sim_dt": alignment["control"]["sim_dt"],
    "decimation": alignment["control"]["decimation"],
    "torso_body": alignment["bodies"]["torso"],
    "root_body": alignment["bodies"]["root"],
    "files": files,
    "motion": motion_meta,
    "rewards": rewards_meta,
    "checkpoints": checkpoints,
    "default_checkpoint": default_ckpt,
  }
  if command_limits is not None:
    manifest["command_limits"] = command_limits
  if profile == "davis":
    depth = (alignment.get("observations") or {}).get("actor", {}).get("depth") or {}
    manifest["obs_inputs"] = ["proprio", "depth"]
    manifest["depth"] = depth
    manifest["proprio_dim"] = packed_obs_dim
  (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  return manifest


def discover_runs(logs_root: Path, task_substr: str) -> list[Path]:
  """Discover runs whose deployment_alignment meta.task_id contains task_substr."""
  runs: list[Path] = []
  needle = task_substr.strip()
  if not needle:
    raise ValueError("--task is required for discovery (do not mix task families)")
  for alignment in logs_root.glob("**/params/deployment_alignment.yaml"):
    run_dir = alignment.parent.parent
    if _find_onnx(run_dir) is None and not _list_pt_checkpoints(run_dir):
      continue
    try:
      raw = _load_alignment(alignment)
    except Exception:
      continue
    tid = _task_id_of(raw)
    if needle not in tid:
      continue
    runs.append(run_dir)
  return sorted(runs)


def write_index(out_root: Path, manifests: list[dict[str, Any]]) -> None:
  out_root.mkdir(parents=True, exist_ok=True)
  index = {
    "packs": [
      {
        "id": m["id"],
        "task_id": m.get("task_id"),
        "play_profile": m.get("play_profile"),
        "obs_dim": m.get("obs_dim"),
        "num_actions": m.get("num_actions"),
        "path": m["id"],
      }
      for m in manifests
    ]
  }
  (out_root / "index.json").write_text(json.dumps(index, indent=2))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", type=Path, help="Single training run directory")
  parser.add_argument(
    "--all-logs",
    action="store_true",
    help=f"Pack eligible runs under {DEFAULT_LOGS} (requires --task)",
  )
  parser.add_argument(
    "--task",
    type=str,
    default="",
    help="Required for --all-logs: substring of meta.task_id "
    "(e.g. AMP-Walk-Run-Wbot-20DOF). Keeps task families separate like Viser.",
  )
  parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--limit", type=int, default=0, help="Max runs for --all-logs")
  parser.add_argument(
    "--replace-index",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Rebuild index only from packs produced this invocation and prune "
    "other packs under --out (default: True). Use --no-replace-index to merge.",
  )
  args = parser.parse_args()

  if not args.run and not args.all_logs:
    parser.error("Provide --run or --all-logs")
  if args.all_logs and not args.task:
    parser.error(
      "--all-logs requires --task <task_id substring>. "
      "Example: --all-logs --task AMP-Walk-Run-Wbot-20DOF\n"
      "Do not pack Kick/Suspended/Getup into the same index as AMP walk."
    )

  runs: list[Path] = []
  if args.run:
    runs.append(args.run)
  if args.all_logs:
    runs.extend(discover_runs(args.logs_root, args.task))

  seen: set[Path] = set()
  uniq: list[Path] = []
  for r in runs:
    rp = r.resolve()
    if rp not in seen:
      seen.add(rp)
      uniq.append(rp)
  if args.limit > 0:
    uniq = uniq[: args.limit]

  manifests: list[dict[str, Any]] = []
  failed: list[str] = []
  for run in uniq:
    try:
      m = pack_run(run, args.out)
      # Optional extra filter when --run without --task
      if args.task and args.task not in str(m.get("task_id") or ""):
        shutil.rmtree(args.out / m["id"], ignore_errors=True)
        raise ValueError(
          f"task_id {m.get('task_id')!r} does not contain --task={args.task!r}"
        )
      manifests.append(m)
      print(
        f"[OK] {m['id']}  task={m.get('task_id')}  profile={m.get('play_profile')}  "
        f"robot={m.get('robot_xml')}  obs={m['obs_dim']} act={m['num_actions']}"
      )
    except Exception as exc:  # noqa: BLE001
      failed.append(f"{run}: {exc}")
      print(f"[SKIP] {run.name}: {exc}", file=sys.stderr)

  # Default: only the packed run(s) appear in the web Pack dropdown.
  if args.replace_index:
    ok_ids = {m["id"] for m in manifests}
    for pack_dir in list(args.out.glob("*")):
      if not pack_dir.is_dir():
        continue
      if pack_dir.name not in ok_ids:
        shutil.rmtree(pack_dir)
        print(f"[PRUNE] {pack_dir.name}")
    write_index(args.out, manifests)
    all_manifests = manifests
  else:
    all_manifests = []
    for pack_dir in sorted(args.out.glob("*/manifest.json")):
      all_manifests.append(json.loads(pack_dir.read_text()))
    write_index(args.out, all_manifests)

  print(
    f"Packed {len(manifests)} run(s); index has {len(all_manifests)}; "
    f"skipped {len(failed)}"
  )
  return 1 if failed and not manifests else 0


if __name__ == "__main__":
  raise SystemExit(main())
