#!/usr/bin/env python3
"""Tune WBOT PD via full-body joint DE with left–right symmetry.

Optimizes log(kp), log(kd) per *symmetric joint group* (left/right share gains).
Each group is boxed to [lo,hi]×seed around nominal kp0/kd0 (default 0.25×–5×).
Candidates evaluated in parallel: each env one PD + a different gait speed.
Interval videos: multi-env speed grid with the shared PD.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from scipy.optimize import differential_evolution
from torch.utils.tensorboard import SummaryWriter

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.wbot.wbot_20dof_constants import (
  ARMATURE_E2806,
  ARMATURE_E4310,
  ARMATURE_E4315,
  ARMATURE_EL05,
  ARMATURE_RS00,
)
from src.tasks.amp.wbot_constants import WBOT_JOINT_NAMES
from src.tasks.suspended.config.wbot_20dof.env_cfgs import (
  wbot_20dof_suspended_env_cfg,
  wbot_20dof_suspended_robot_cfg,
)
from src.tasks.suspended.mdp.reference import (
  ARM_SWINGS,
  LEG_AMPLITUDES,
  LEG_PHASES,
)

# Seed PD (= robot constants). DE searches log(kp)/log(kd) inside mult boxes.
_FAMILY = {
  # Seeds match booster_gym T1 PD (see wbot_20dof_constants.py).
  "shoulder_pitch": dict(armature=ARMATURE_RS00, kp0=20.0, kd0=0.50),
  "shoulder_roll": dict(armature=ARMATURE_RS00, kp0=20.0, kd0=0.50),
  "shoulder_yaw": dict(armature=ARMATURE_EL05, kp0=20.0, kd0=0.50),
  "elbow": dict(armature=ARMATURE_EL05, kp0=20.0, kd0=0.50),
  "ankle_roll": dict(armature=ARMATURE_E2806, kp0=50.0, kd0=1.0),
  "hip_yaw": dict(armature=ARMATURE_E4310, kp0=200.0, kd0=5.0),
  "ankle_pitch": dict(armature=ARMATURE_E4310, kp0=50.0, kd0=1.0),
  "hip_pitch": dict(armature=ARMATURE_E4315, kp0=200.0, kd0=5.0),
  "hip_roll": dict(armature=ARMATURE_E4315, kp0=200.0, kd0=5.0),
  "knee": dict(armature=ARMATURE_E4315, kp0=200.0, kd0=5.0),
}

# Default search box: seed × [lo, hi]. Keeps gains deployable (no kp>1000).
_KP_MULT = (0.25, 5.0)  # e.g. hip_pitch kp ∈ [19.4, 387.5]
_KD_MULT = (0.25, 5.0)  # e.g. hip_pitch kd ∈ [0.52, 10.35]


def _family_key(joint: str) -> str:
  for key in _FAMILY:
    if key in joint:
      return key
  raise KeyError(joint)


def _tb_tag(joint: str) -> str:
  return re.sub(r"_joint$", "", joint)


def _side_stem(joint: str) -> str:
  """left_hip_pitch_joint → hip_pitch."""
  name = re.sub(r"_joint$", "", joint)
  name = re.sub(r"^(left|right)_", "", name)
  return name


def build_joints(names: tuple[str, ...] | None = None) -> list[dict]:
  out = []
  for jn in names or WBOT_JOINT_NAMES:
    fam = _FAMILY[_family_key(jn)]
    out.append(
      {
        "name": jn,
        "tag": _tb_tag(jn),
        "stem": _side_stem(jn),
        "armature": float(fam["armature"]),
        "kp0": float(fam["kp0"]),
        "kd0": float(fam["kd0"]),
      }
    )
  return out


def build_sym_groups(joints: list[dict]) -> list[dict]:
  """Group left/right joints that must share identical PD."""
  order: list[str] = []
  buckets: dict[str, list[dict]] = {}
  for j in joints:
    stem = j["stem"]
    if stem not in buckets:
      order.append(stem)
      buckets[stem] = []
    buckets[stem].append(j)
  groups = []
  for stem in order:
    members = buckets[stem]
    if len(members) not in (1, 2):
      raise RuntimeError(f"unexpected symmetry group {stem}: {[m['name'] for m in members]}")
    groups.append(
      {
        "stem": stem,
        "tag": stem,
        "members": [m["name"] for m in members],
        "member_indices": [joints.index(m) for m in members],
        "armature": float(members[0]["armature"]),
        "kp0": float(members[0]["kp0"]),
        "kd0": float(members[0]["kd0"]),
      }
    )
  return groups


def build_group_log_bounds(
  groups: list[dict],
  kp_mult: tuple[float, float] = _KP_MULT,
  kd_mult: tuple[float, float] = _KD_MULT,
) -> list[tuple[float, float]]:
  """Per-group log(kp)/log(kd) bounds from seed × [lo, hi]."""
  kp_lo, kp_hi = float(kp_mult[0]), float(kp_mult[1])
  kd_lo, kd_hi = float(kd_mult[0]), float(kd_mult[1])
  if not (0.0 < kp_lo < kp_hi and 0.0 < kd_lo < kd_hi):
    raise ValueError(f"bad mult ranges kp={kp_mult} kd={kd_mult}")
  bounds: list[tuple[float, float]] = []
  for g in groups:
    kp0 = max(float(g["kp0"]), 1e-6)
    kd0 = max(float(g["kd0"]), 1e-6)
    bounds.append((math.log(kp0 * kp_lo), math.log(kp0 * kp_hi)))
    bounds.append((math.log(kd0 * kd_lo), math.log(kd0 * kd_hi)))
  return bounds


def _clip_x_to_bounds(x: list[float] | np.ndarray, bounds: list[tuple[float, float]]) -> list[float]:
  out = []
  for xi, (lo, hi) in zip(x, bounds, strict=True):
    out.append(float(np.clip(float(xi), lo, hi)))
  return out


def _make_env(
  num_envs: int,
  device: str,
  *,
  render_mode: str | None = None,
  viewer_wh: tuple[int, int] = (640, 360),
) -> ManagerBasedRlEnv:
  cfg = wbot_20dof_suspended_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1.0e9
  cfg.observations["actor"].enable_corruption = False
  cfg.terminations = {}
  cfg.curriculum = {}
  cfg.rewards = {}
  cfg.metrics = {}
  if hasattr(cfg, "viewer") and cfg.viewer is not None:
    cfg.viewer.width = viewer_wh[0]
    cfg.viewer.height = viewer_wh[1]
    # Per-env camera tiles; do not overlay other envs (spacing is 0).
    cfg.viewer.max_extra_envs = 0
  robot = wbot_20dof_suspended_robot_cfg()
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(
      replace(
        a,
        delay_min_lag=0,
        delay_max_lag=0,
        delay_hold_prob=0.0,
        delay_update_period=0,
      )
      for a in robot.articulation.actuators
    ),
  )
  cfg.scene.entities = {"robot": robot}
  cfg.events["pd_gains_expand"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (1.0, 1.0),
      "kd_range": (1.0, 1.0),
      "operation": "scale",
    },
  )
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)


def _render_env_frame(env: ManagerBasedRlEnv, env_idx: int) -> np.ndarray | None:
  """Render a single env index (suspended spacing is 0, so no multi-geom overlay)."""
  renderer = getattr(env, "_offline_renderer", None)
  if renderer is None:
    return None
  env_idx = int(max(0, min(env_idx, env.num_envs - 1)))
  env.cfg.viewer.env_idx = env_idx
  renderer._cfg.env_idx = env_idx
  renderer._extra_env_ids = []
  frame = env.render()
  return None if frame is None else np.asarray(frame)


def _annotate_tile(frame: np.ndarray, text: str) -> np.ndarray:
  out = np.array(frame, copy=True)
  try:
    import cv2

    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
      out,
      text,
      (8, 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )
  except Exception:
    pass
  return out


def _tile_grid(tiles: list[np.ndarray], ncols: int) -> np.ndarray:
  if not tiles:
    raise ValueError("no tiles")
  h, w = tiles[0].shape[:2]
  ncols = max(1, min(ncols, len(tiles)))
  nrows = (len(tiles) + ncols - 1) // ncols
  canvas = np.zeros((nrows * h, ncols * w, 3), dtype=tiles[0].dtype)
  for i, tile in enumerate(tiles):
    r, c = divmod(i, ncols)
    canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
  return canvas


def _joint_ctrl_map(env: ManagerBasedRlEnv) -> dict[str, int]:
  m: dict[str, int] = {}
  for act in env.scene["robot"].actuators:
    for name, cid in zip(act.target_names, act.ctrl_ids.tolist()):
      m[str(name)] = int(cid)
  return m


def _set_joint_pd(
  env: ManagerBasedRlEnv,
  ctrl_id: int,
  kp: torch.Tensor,
  kd: torch.Tensor,
) -> None:
  n = env.num_envs
  env_ids = torch.arange(n, device=env.device, dtype=torch.int)
  env.sim.model.actuator_gainprm[env_ids, ctrl_id, 0] = kp
  env.sim.model.actuator_biasprm[env_ids, ctrl_id, 1] = -kp
  env.sim.model.actuator_biasprm[env_ids, ctrl_id, 2] = -kd


def _apply_pd_matrix(
  env: ManagerBasedRlEnv,
  joints: list[dict],
  ctrl_map: dict[str, int],
  kp_mat: torch.Tensor,
  kd_mat: torch.Tensor,
) -> None:
  """kp_mat/kd_mat: (num_envs, n_joints) — column j = joint j."""
  for ji, j in enumerate(joints):
    _set_joint_pd(env, ctrl_map[j["name"]], kp_mat[:, ji], kd_mat[:, ji])


def _ref_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=WBOT_JOINT_NAMES, preserve_order=True)


def make_env_periods(
  num_envs: int,
  period_slow: float,
  period_fast: float,
  device: str | torch.device,
) -> torch.Tensor:
  """One unique log-spaced gait period per env (slow→fast)."""
  return torch.logspace(
    math.log10(period_slow),
    math.log10(period_fast),
    num_envs,
    device=device,
    dtype=torch.float32,
  )


def _amp_scale_from_period(
  periods: torch.Tensor,
  period_slow: float,
  period_fast: float,
  *,
  base_amp: float = 1.0,
) -> torch.Tensor:
  """Faster gait → slightly larger swing (base_amp slow … base_amp*1.35 fast)."""
  t = (math.log(period_slow) - torch.log(periods)) / max(
    math.log(period_slow) - math.log(period_fast), 1e-6
  )
  return float(base_amp) * (1.0 + 0.35 * t.clamp(0, 1))


def reference_multispeed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  periods: torch.Tensor,
  *,
  amp_scale: torch.Tensor | None = None,
  phase_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Walk reference with per-env period (and optional amp scale)."""
  robot = env.scene[asset_cfg.name]
  omega = (2 * math.pi) / periods.clamp_min(1e-3)
  if phase_override is None:
    t = env.episode_length_buf.float() * env.step_dt
    p = t * omega
  else:
    p = phase_override
  if amp_scale is None:
    amp_scale = torch.ones(env.num_envs, device=env.device)
  q = robot.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
  dq = torch.zeros_like(q)
  for start, offset in LEG_PHASES:
    s, c = torch.sin(p + offset), torch.cos(p + offset)
    for index, amplitude in LEG_AMPLITUDES:
      a = amplitude * amp_scale
      q[:, start + index] += a * s
      dq[:, start + index] = a * omega * c
  joint_names = tuple(asset_cfg.joint_names or ())
  name_to_local = {name: i for i, name in enumerate(joint_names)}
  for name, offset, amplitude in ARM_SWINGS:
    local = name_to_local.get(name)
    if local is None:
      continue
    s, c = torch.sin(p + offset), torch.cos(p + offset)
    a = amplitude * amp_scale
    q[:, local] += a * s
    dq[:, local] = a * omega * c
  return q, dq


def _action_track_reference(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  periods: torch.Tensor,
  amp_scale: torch.Tensor | None = None,
  phase_override: torch.Tensor | None = None,
) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  q_ref, _ = reference_multispeed(
    env, asset_cfg, periods, amp_scale=amp_scale, phase_override=phase_override
  )
  robot = env.scene["robot"]
  name_to_q = {n: i for i, n in enumerate(WBOT_JOINT_NAMES)}
  tids = term._target_ids
  if torch.is_tensor(tids):
    tids = tids.tolist()
  action = torch.zeros(env.num_envs, term.action_dim, device=env.device)
  scale, offset = term._scale, term._offset
  for ai, jid in enumerate(tids):
    jn = robot.joint_names[int(jid)]
    li = name_to_q[jn]
    action[:, ai] = (q_ref[:, li] - offset[:, ai]) / scale[:, ai].clamp_min(1e-6)
  return action


def run_walk_eval(
  env: ManagerBasedRlEnv,
  joints: list[dict],
  duration_s: float,
  noise_std: float,
  periods: torch.Tensor,
  period_slow: float,
  period_fast: float,
  *,
  ref_amp: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Whole-body walk track; also returns per-joint RMSE (num_envs, n_joints)."""
  asset_cfg = _ref_asset_cfg()
  asset_cfg.resolve(env.scene)
  robot = env.scene["robot"]
  jids = [list(robot.joint_names).index(j["name"]) for j in joints]
  ref_is = [WBOT_JOINT_NAMES.index(j["name"]) for j in joints]
  n_j = len(joints)
  amp_scale = _amp_scale_from_period(
    periods, period_slow, period_fast, base_amp=ref_amp
  )

  env.reset()
  n_steps = int(round(duration_s / env.step_dt))
  warmup = int(round(0.5 * float(period_slow) / env.step_dt))
  err2_j = torch.zeros(env.num_envs, n_j, device=env.device)
  err2_all = torch.zeros(env.num_envs, device=env.device)
  osc = torch.zeros(env.num_envs, device=env.device)
  prev_e = torch.zeros(env.num_envs, device=env.device)
  n_acc = 0

  for k in range(n_steps):
    action = _action_track_reference(env, asset_cfg, periods, amp_scale)
    env.step(action)
    if k < warmup:
      continue
    q_ref, _ = reference_multispeed(env, asset_cfg, periods, amp_scale=amp_scale)
    q = robot.data.joint_pos
    if noise_std > 0:
      q = q + noise_std * torch.randn_like(q)
    e_j = q[:, jids] - q_ref[:, ref_is]
    err2_j += e_j.pow(2)
    err2_all += (robot.data.joint_pos[:, asset_cfg.joint_ids] - q_ref).pow(2).mean(dim=-1)
    em = e_j.mean(dim=-1)
    osc += ((em * prev_e < 0) & (em.abs() > 0.03)).float()
    prev_e = em
    n_acc += 1

  rmse_j = torch.sqrt(err2_j / max(n_acc, 1))
  return {
    "rmse_j": rmse_j,
    "rmse": torch.sqrt(rmse_j.pow(2).mean(dim=-1)),
    "rmse_all": torch.sqrt(err2_all / max(n_acc, 1)),
    "osc": osc / max(n_acc, 1),
  }


def _zeta_penalty(
  kp_mat: torch.Tensor,
  kd_mat: torch.Tensor,
  armatures: torch.Tensor,
) -> torch.Tensor:
  """Mean zeta-bound violation over joints. kp/kd (N,J), armature (J,)."""
  i = armatures[None, :].clamp_min(1e-12)
  kd_lo = 2.0 * 0.5 * torch.sqrt(kp_mat * i)
  kd_hi = 2.0 * 2.5 * torch.sqrt(kp_mat * i)
  pen = (kd_lo - kd_mat).clamp_min(0) / kd_lo.clamp_min(1e-6) + (
    kd_mat - kd_hi
  ).clamp_min(0) / kd_hi.clamp_min(1e-6)
  return pen.mean(dim=-1)


def _cost_from_metrics(
  metrics: dict[str, torch.Tensor],
  kp_mat: torch.Tensor,
  kd_mat: torch.Tensor,
  armatures: torch.Tensor,
) -> torch.Tensor:
  return (
    5.0 * metrics["rmse"]
    + 1.0 * metrics["rmse_all"]
    + 2.0 * metrics["osc"]
    + 0.3 * _zeta_penalty(kp_mat, kd_mat, armatures)
  )


def _x_to_group_kp_kd(x: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
  """x = interleaved log_kp/log_kd per symmetry group → (S, G)."""
  if x.ndim == 1:
    x = x.reshape(-1, 1)
  assert x.shape[0] == 2 * n_groups
  log_kp = np.asarray(x[0::2], dtype=np.float64)
  log_kd = np.asarray(x[1::2], dtype=np.float64)
  kp = np.clip(np.exp(log_kp), 1e-4, 1e5).T
  kd = np.clip(np.exp(log_kd), 1e-5, 1e4).T
  return kp, kd


def _expand_group_pd(
  kp_g: np.ndarray,
  kd_g: np.ndarray,
  groups: list[dict],
  n_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Broadcast group gains to all joints. kp_g/kd_g: (S, G) → (S, J)."""
  s = kp_g.shape[0]
  kp = np.zeros((s, n_joints), dtype=np.float64)
  kd = np.zeros((s, n_joints), dtype=np.float64)
  for gi, g in enumerate(groups):
    for ji in g["member_indices"]:
      kp[:, ji] = kp_g[:, gi]
      kd[:, ji] = kd_g[:, gi]
  return kp, kd


def _x_to_kp_kd(
  x: np.ndarray, groups: list[dict], n_joints: int
) -> tuple[np.ndarray, np.ndarray]:
  """Search x (2*G,) → full-body kp/kd (S, J) with L/R tied."""
  kp_g, kd_g = _x_to_group_kp_kd(x, len(groups))
  return _expand_group_pd(kp_g, kd_g, groups, n_joints)


def evaluate_candidates(
  env: ManagerBasedRlEnv,
  joints: list[dict],
  ctrl_map: dict[str, int],
  kp_np: np.ndarray,
  kd_np: np.ndarray,
  armatures: torch.Tensor,
  duration_s: float,
  noise_std: float,
  periods: torch.Tensor,
  period_slow: float,
  period_fast: float,
  *,
  ref_amp: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Parallel eval: env i runs candidate i with that env's gait speed."""
  s = int(kp_np.shape[0])
  n_j = len(joints)
  out = np.empty(s, dtype=np.float64)
  rmse = np.empty(s, dtype=np.float64)
  rmse_all = np.empty(s, dtype=np.float64)
  rmse_j = np.empty((s, n_j), dtype=np.float64)
  osc = np.empty(s, dtype=np.float64)

  for start in range(0, s, env.num_envs):
    end = min(start + env.num_envs, s)
    chunk = end - start
    kp_mat = torch.zeros(env.num_envs, n_j, device=env.device)
    kd_mat = torch.zeros(env.num_envs, n_j, device=env.device)
    kp_mat[:chunk] = torch.as_tensor(kp_np[start:end], device=env.device, dtype=torch.float32)
    kd_mat[:chunk] = torch.as_tensor(kd_np[start:end], device=env.device, dtype=torch.float32)
    if chunk < env.num_envs:
      kp_mat[chunk:] = kp_mat[0]
      kd_mat[chunk:] = kd_mat[0]
    _apply_pd_matrix(env, joints, ctrl_map, kp_mat, kd_mat)
    metrics = run_walk_eval(
      env,
      joints,
      duration_s,
      noise_std,
      periods,
      period_slow,
      period_fast,
      ref_amp=ref_amp,
    )
    cost = _cost_from_metrics(metrics, kp_mat, kd_mat, armatures)
    cost_np = cost[:chunk].detach().float().cpu().numpy()
    bad = ~np.isfinite(cost_np)
    if bad.any():
      cost_np = cost_np.copy()
      cost_np[bad] = 1.0e3
    out[start:end] = cost_np
    rmse_np = metrics["rmse"][:chunk].detach().float().cpu().numpy()
    rmse_all_np = metrics["rmse_all"][:chunk].detach().float().cpu().numpy()
    rmse_j_np = metrics["rmse_j"][:chunk].detach().float().cpu().numpy()
    osc_np = metrics["osc"][:chunk].detach().float().cpu().numpy()
    for arr in (rmse_np, rmse_all_np, osc_np):
      arr[~np.isfinite(arr)] = 1.0e3
    rmse_j_np[~np.isfinite(rmse_j_np)] = 1.0e3
    rmse[start:end] = rmse_np
    rmse_all[start:end] = rmse_all_np
    rmse_j[start:end] = rmse_j_np
    osc[start:end] = osc_np
  return out, {"rmse": rmse, "rmse_all": rmse_all, "rmse_j": rmse_j, "osc": osc}

def decode_baseline(
  x: np.ndarray, groups: list[dict], joints: list[dict]
) -> dict[str, list[float]]:
  kp, kd = _x_to_kp_kd(np.asarray(x, dtype=np.float64), groups, len(joints))
  return {
    j["name"]: [round(float(kp[0, i]), 3), round(float(kd[0, i]), 3)]
    for i, j in enumerate(joints)
  }


def _warmstart_group_logs(
  groups: list[dict], prev_final: dict[str, list[float]]
) -> list[float]:
  """Average L/R gains from a prior (possibly asymmetric) final map → log x0."""
  x0: list[float] = []
  for g in groups:
    kps, kds = [], []
    for name in g["members"]:
      if name in prev_final:
        kps.append(float(prev_final[name][0]))
        kds.append(float(prev_final[name][1]))
    if not kps:
      kps = [g["kp0"]]
      kds = [g["kd0"]]
    kp = float(np.mean(kps))
    kd = float(np.mean(kds))
    x0.append(float(np.log(max(kp, 1e-6))))
    x0.append(float(np.log(max(kd, 1e-6))))
  return x0


_CURVE_JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
)


def _save_response_curves(
  t: np.ndarray,
  q: np.ndarray,
  q_ref: np.ndarray,
  periods: np.ndarray,
  joint_names: list[str],
  out_dir: Path,
  *,
  gen: int,
  writer: SummaryWriter | None = None,
) -> list[Path]:
  """q/q_ref: (T, n_envs, n_joints). Save per-speed panels + overview overlays."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  out_dir.mkdir(parents=True, exist_ok=True)
  name_to_i = {n: i for i, n in enumerate(joint_names)}
  plot_ids = [(n, name_to_i[n]) for n in _CURVE_JOINTS if n in name_to_i]
  if not plot_ids:
    return []

  n_envs = q.shape[1]
  saved: list[Path] = []

  # Per-speed multi-joint response.
  ncols = 5
  nrows = int(math.ceil(len(plot_ids) / ncols))
  for ei in range(n_envs):
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    T = float(periods[ei])
    fig.suptitle(
      f"gen={gen}  env={ei}  T={T:.2f}s  f={1.0 / T:.2f}Hz  (q vs q_ref)",
      fontsize=11,
    )
    for ax, (jn, ji) in zip(axes_flat, plot_ids):
      ax.plot(t, q_ref[:, ei, ji], "k--", lw=1.0, label="ref")
      ax.plot(t, q[:, ei, ji], "C0-", lw=1.2, label="meas")
      err = q[:, ei, ji] - q_ref[:, ei, ji]
      rmse = float(np.sqrt(np.mean(err**2)))
      ax.set_title(f"{_tb_tag(jn)}  RMSE={rmse:.3f}", fontsize=9)
      ax.grid(True, alpha=0.3)
      ax.tick_params(labelsize=8)
    for ax in axes_flat[len(plot_ids) :]:
      ax.axis("off")
    axes_flat[0].legend(loc="upper right", fontsize=7)
    for ax in axes_flat[ncols * (nrows - 1) :]:
      ax.set_xlabel("t (s)", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"gen{gen:03d}_speed{ei:02d}_T{T:.2f}s.png"
    fig.savefig(path, dpi=120)
    if writer is not None:
      fig.canvas.draw()
      img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
      writer.add_image(
        f"curves/speed_{ei:02d}", img, global_step=gen, dataformats="HWC"
      )
    plt.close(fig)
    saved.append(path)

  # Overview: overlay all speeds for a few joints.
  overview = [
    n
    for n in (
      "left_hip_pitch_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_shoulder_pitch_joint",
    )
    if n in name_to_i
  ]
  if overview:
    fig, axes = plt.subplots(len(overview), 1, figsize=(9, 2.2 * len(overview)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"gen={gen}  multi-speed overlay (meas solid, ref dashed)", fontsize=11)
    cmap = plt.get_cmap("viridis")
    for ax, jn in zip(axes, overview):
      ji = name_to_i[jn]
      for ei in range(n_envs):
        color = cmap(ei / max(n_envs - 1, 1))
        T = float(periods[ei])
        ax.plot(t, q_ref[:, ei, ji], ls="--", color=color, lw=0.8, alpha=0.7)
        ax.plot(
          t,
          q[:, ei, ji],
          ls="-",
          color=color,
          lw=1.0,
          label=f"T={T:.2f}s",
        )
      ax.set_ylabel(_tb_tag(jn), fontsize=9)
      ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=min(4, n_envs), fontsize=7, loc="upper right")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    path = out_dir / f"gen{gen:03d}_overview_multispeed.png"
    fig.savefig(path, dpi=120)
    if writer is not None:
      fig.canvas.draw()
      img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
      writer.add_image("curves/overview_multispeed", img, global_step=gen, dataformats="HWC")
    plt.close(fig)
    saved.append(path)

  print(f"[pd-tune] response curves gen={gen} → {out_dir} ({len(saved)} figs)", flush=True)
  return saved


def record_test_video(
  video_env: ManagerBasedRlEnv,
  joints: list[dict],
  ctrl_map: dict[str, int],
  baseline: dict[str, list[float]],
  video_dir: Path,
  duration_s: float,
  period_slow: float,
  period_fast: float,
  writer: SummaryWriter | None = None,
  *,
  gen: int = 0,
  name_prefix: str = "walk_track",
  grid_cols: int | None = None,
  ref_amp: float = 1.0,
) -> Path:
  """Record multi-env grid mp4 + tracking response curves (same PD, varied speeds)."""
  import mediapy as media

  video_dir.mkdir(parents=True, exist_ok=True)
  curves_dir = video_dir.parent / "curves"
  n = video_env.num_envs
  n_j = len(joints)
  periods = make_env_periods(n, period_slow, period_fast, video_env.device)
  amp = _amp_scale_from_period(
    periods, period_slow, period_fast, base_amp=ref_amp
  )
  kp_row = torch.tensor(
    [baseline[j["name"]][0] for j in joints], device=video_env.device
  )
  kd_row = torch.tensor(
    [baseline[j["name"]][1] for j in joints], device=video_env.device
  )
  kp_mat = kp_row[None, :].expand(n, n_j).contiguous()
  kd_mat = kd_row[None, :].expand(n, n_j).contiguous()
  _apply_pd_matrix(video_env, joints, ctrl_map, kp_mat, kd_mat)

  n_steps = int(round(duration_s / video_env.step_dt))
  prefix = f"{name_prefix}_gen{gen:03d}"
  out_path = video_dir / f"{prefix}_speeds{n}.mp4"
  asset_cfg = _ref_asset_cfg()
  asset_cfg.resolve(video_env.scene)
  robot = video_env.scene["robot"]
  jids = [list(robot.joint_names).index(j["name"]) for j in joints]
  ref_is = [WBOT_JOINT_NAMES.index(j["name"]) for j in joints]
  video_env.reset()

  phase = torch.zeros(n, device=video_env.device)
  labels = [
    f"T={float(periods[i]):.2f}s  f={1.0 / float(periods[i]):.2f}Hz"
    for i in range(n)
  ]
  ncols = grid_cols if grid_cols is not None else int(math.ceil(math.sqrt(n)))
  frames: list[np.ndarray] = []
  fps = max(1, int(round(1.0 / video_env.step_dt)))
  q_hist: list[np.ndarray] = []
  qref_hist: list[np.ndarray] = []
  t_hist: list[float] = []

  for k in range(n_steps):
    action = _action_track_reference(
      video_env, asset_cfg, periods, amp, phase_override=phase
    )
    video_env.step(action)
    omega = (2 * math.pi) / periods.clamp_min(1e-3)
    phase = phase + omega * video_env.step_dt
    q_ref, _ = reference_multispeed(
      video_env, asset_cfg, periods, amp_scale=amp, phase_override=phase
    )
    q_now = robot.data.joint_pos[:, jids].detach().float().cpu().numpy()
    qref_now = q_ref[:, ref_is].detach().float().cpu().numpy()
    q_hist.append(q_now)
    qref_hist.append(qref_now)
    t_hist.append((k + 1) * float(video_env.step_dt))
    tiles = []
    for ei in range(n):
      frame = _render_env_frame(video_env, ei)
      if frame is None:
        continue
      tiles.append(_annotate_tile(frame, labels[ei]))
    if tiles:
      frames.append(_tile_grid(tiles, ncols))

  # Response curves (even if render failed).
  t_np = np.asarray(t_hist, dtype=np.float64)
  q_np = np.stack(q_hist, axis=0)  # T,E,J
  qref_np = np.stack(qref_hist, axis=0)
  _save_response_curves(
    t_np,
    q_np,
    qref_np,
    periods.detach().float().cpu().numpy(),
    [j["name"] for j in joints],
    curves_dir,
    gen=gen,
    writer=writer,
  )

  if not frames:
    raise RuntimeError("multi-speed video produced no frames")
  media.write_video(str(out_path), frames, fps=fps)
  print(
    f"[pd-tune] interval video gen={gen}  envs={n} speeds  → {out_path}",
    flush=True,
  )
  if writer is not None:
    step = max(1, len(frames) // 120)
    clip = np.stack(frames[::step][:120], axis=0)
    if clip.ndim == 4 and clip.shape[-1] == 3:
      clip = clip.transpose(0, 3, 1, 2)
      try:
        writer.add_video("test/walk_track_multspeed", clip[None], global_step=gen, fps=fps)
      except Exception as e:
        print(f"[pd-tune] TB video skipped: {e}", flush=True)
  return out_path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--log-root", type=Path, default=Path("logs/pd_tune"))
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--duration", type=float, default=4.0)
  parser.add_argument("--noise-std", type=float, default=0.008)
  parser.add_argument(
    "--popsize",
    type=int,
    default=None,
    help="DE popsize multiplier (pop=popsize*n_params). Default ≈ num_envs/n_params",
  )
  parser.add_argument("--maxiter", type=int, default=5000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--joints", nargs="*", default=None)
  parser.add_argument("--video-duration", type=float, default=4.0)
  parser.add_argument(
    "--video-envs",
    type=int,
    default=8,
    help="Parallel envs in interval video (same PD, different speeds)",
  )
  parser.add_argument(
    "--video-tile-cols",
    type=int,
    default=4,
    help="Grid columns for multi-speed video tiles",
  )
  parser.add_argument(
    "--video-interval",
    type=int,
    default=10,
    help="Record walk-track video every N DE generations (0=disable)",
  )
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument(
    "--init-json",
    type=Path,
    default=None,
    help="Warm-start x0 from a prior best_pd.json (uses 'final' map)",
  )
  parser.add_argument(
    "--early-stop-patience",
    type=int,
    default=40,
    help="Stop if running-best cost does not improve for this many gens (0=off)",
  )
  parser.add_argument(
    "--early-stop-delta",
    type=float,
    default=0.001,
    help="Minimum absolute cost drop to count as improvement",
  )
  parser.add_argument(
    "--early-stop-cost",
    type=float,
    default=None,
    help="Optional absolute cost target; stop when running-best <= this",
  )
  parser.add_argument(
    "--period-slow",
    type=float,
    default=2.0,
    help="Slowest gait period (s); each env gets a unique log-spaced period",
  )
  parser.add_argument(
    "--period-fast",
    type=float,
    default=0.35,
    help="Fastest gait period (s)",
  )
  parser.add_argument(
    "--kp-mult-lo",
    type=float,
    default=_KP_MULT[0],
    help="Lower kp bound as multiple of seed kp0",
  )
  parser.add_argument(
    "--kp-mult-hi",
    type=float,
    default=_KP_MULT[1],
    help="Upper kp bound as multiple of seed kp0",
  )
  parser.add_argument(
    "--kd-mult-lo",
    type=float,
    default=_KD_MULT[0],
    help="Lower kd bound as multiple of seed kd0",
  )
  parser.add_argument(
    "--kd-mult-hi",
    type=float,
    default=_KD_MULT[1],
    help="Upper kd bound as multiple of seed kd0",
  )
  parser.add_argument(
    "--ref-amp",
    type=float,
    default=2.0,
    help="Base walk reference amplitude scale (1.0=LEG/ARM defaults; 2.0=2×)",
  )
  args = parser.parse_args()

  joints = build_joints(tuple(args.joints) if args.joints else None)
  groups = build_sym_groups(joints)
  n_j = len(joints)
  n_g = len(groups)
  n_params = 2 * n_g  # left/right tied → one (log_kp, log_kd) per stem
  popsize = args.popsize if args.popsize is not None else max(5, args.num_envs // n_params)
  pop_n = max(5, popsize * n_params)

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = args.log_root / f"{stamp}_fullbody_pd_de"
  log_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(log_dir=str(log_dir))

  kp_mult = (float(args.kp_mult_lo), float(args.kp_mult_hi))
  kd_mult = (float(args.kd_mult_lo), float(args.kd_mult_hi))
  ref_amp = float(args.ref_amp)
  print(
    f"[pd-tune] FULL-BODY joint DE  dims={n_params} "
    f"({n_g} L/R-symmetric groups×2, {n_j} joints)  "
    f"envs={args.num_envs} PARALLEL (1 PD/env + unique speed)  "
    f"period {args.period_slow:.2f}s→{args.period_fast:.2f}s  "
    f"ref_amp={ref_amp:.2f}×  "
    f"PD=exp(log) L=R  kp∈[{kp_mult[0]:.2f},{kp_mult[1]:.2f}]×seed  "
    f"kd∈[{kd_mult[0]:.2f},{kd_mult[1]:.2f}]×seed  "
    f"pop≈{pop_n} (popsize={popsize})  "
    f"maxiter={args.maxiter}  duration={args.duration}s  noise={args.noise_std}  "
    f"early_stop patience={args.early_stop_patience} delta={args.early_stop_delta} "
    f"target={args.early_stop_cost}  log={log_dir}",
    flush=True,
  )
  for g in groups:
    print(
      f"  group {g['stem']}: {g['members']}  "
      f"kp∈[{g['kp0']*kp_mult[0]:.1f},{g['kp0']*kp_mult[1]:.1f}]  "
      f"kd∈[{g['kd0']*kd_mult[0]:.2f},{g['kd0']*kd_mult[1]:.2f}]",
      flush=True,
    )

  env = _make_env(args.num_envs, args.device)
  ctrl_map = _joint_ctrl_map(env)
  missing = [j["name"] for j in joints if j["name"] not in ctrl_map]
  if missing:
    raise RuntimeError(f"joints missing ctrl ids: {missing}")

  periods = make_env_periods(
    args.num_envs, args.period_slow, args.period_fast, env.device
  )
  print(
    f"[pd-tune] env periods: slow={float(periods[0]):.3f}s  "
    f"mid={float(periods[args.num_envs // 2]):.3f}s  "
    f"fast={float(periods[-1]):.3f}s  (n={args.num_envs})",
    flush=True,
  )

  armatures = torch.tensor(
    [j["armature"] for j in joints], device=env.device, dtype=torch.float32
  )
  # Search variables: log(kp), log(kd) per symmetry group (L=R + seed×mult box).
  bounds = build_group_log_bounds(groups, kp_mult=kp_mult, kd_mult=kd_mult)
  x0 = []
  for g in groups:
    x0.append(float(np.log(max(g["kp0"], 1e-6))))
    x0.append(float(np.log(max(g["kd0"], 1e-6))))
  x0 = _clip_x_to_bounds(x0, bounds)

  if args.init_json is not None:
    prev = json.loads(args.init_json.read_text())["final"]
    x0 = _clip_x_to_bounds(_warmstart_group_logs(groups, prev), bounds)
    print(
      f"[pd-tune] warm-start from {args.init_json} "
      f"(L/R averaged, clipped to seed×mult bounds)",
      flush=True,
    )

  history: list[dict] = []
  videos_recorded: list[str] = []
  gen = {"i": 0}
  best_x = np.asarray(x0, dtype=np.float64)
  # Running global best (noise-robust early stop uses this, not per-gen noise).
  early = {
    "best_cost": float("inf"),
    "best_x": best_x.copy(),
    "stale": 0,
    "reason": None,
  }

  video_env = None
  video_ctrl_map: dict[str, int] | None = None
  video_interval = 0 if args.no_video else max(0, int(args.video_interval))
  if video_interval > 0:
    n_vid = max(1, int(args.video_envs))
    print(
      f"[pd-tune] interval video every {video_interval} gens  "
      f"duration={args.video_duration}s  video_envs={n_vid} "
      f"(shared PD, distinct speeds)",
      flush=True,
    )
    video_env = _make_env(
      n_vid, args.device, render_mode="rgb_array", viewer_wh=(320, 240)
    )
    video_ctrl_map = _joint_ctrl_map(video_env)

  def objective(x: np.ndarray) -> np.ndarray:
    nonlocal best_x
    kp_g, kd_g = _x_to_group_kp_kd(x, n_g)
    kp_np, kd_np = _expand_group_pd(kp_g, kd_g, groups, n_j)
    costs, metrics = evaluate_candidates(
      env,
      joints,
      ctrl_map,
      kp_np,
      kd_np,
      armatures,
      args.duration,
      args.noise_std,
      periods,
      args.period_slow,
      args.period_fast,
      ref_amp=ref_amp,
    )
    best_i = int(np.argmin(costs))
    best_cost = float(costs[best_i])
    # Keep search-space coords (per-group log gains).
    best_x = np.empty(n_params, dtype=np.float64)
    best_x[0::2] = np.log(np.maximum(kp_g[best_i], 1e-12))
    best_x[1::2] = np.log(np.maximum(kd_g[best_i], 1e-12))
    gi = gen["i"]

    improved = best_cost < early["best_cost"] - args.early_stop_delta
    if best_cost < early["best_cost"]:
      early["best_cost"] = best_cost
      early["best_x"] = best_x.copy()
    if improved:
      early["stale"] = 0
    else:
      early["stale"] += 1

    writer.add_scalar("fullbody/gen_best_cost", best_cost, gi)
    writer.add_scalar("fullbody/running_best_cost", early["best_cost"], gi)
    writer.add_scalar("fullbody/early_stale", early["stale"], gi)
    writer.add_scalar("fullbody/gen_mean_cost", float(np.mean(costs)), gi)
    writer.add_scalar("fullbody/gen_std_cost", float(np.std(costs)), gi)
    writer.add_histogram("fullbody/gen_cost_hist", costs, gi)
    writer.add_scalar("fullbody/gen_best_rmse", float(metrics["rmse"][best_i]), gi)
    writer.add_scalar("fullbody/gen_best_rmse_all", float(metrics["rmse_all"][best_i]), gi)
    for gi_g, g in enumerate(groups):
      tag = g["tag"]
      writer.add_scalar(f"group/{tag}/gen_best_kp", float(kp_g[best_i, gi_g]), gi)
      writer.add_scalar(f"group/{tag}/gen_best_kd", float(kd_g[best_i, gi_g]), gi)
      # Mean RMSE over L/R members for this group.
      mi = g["member_indices"]
      writer.add_scalar(
        f"group/{tag}/gen_best_rmse",
        float(metrics["rmse_j"][best_i, mi].mean()),
        gi,
      )
      for ji in mi:
        jtag = joints[ji]["tag"]
        writer.add_scalar(f"joint/{jtag}/gen_best_kp", float(kp_np[best_i, ji]), gi)
        writer.add_scalar(f"joint/{jtag}/gen_best_kd", float(kd_np[best_i, ji]), gi)
        writer.add_scalar(
          f"joint/{jtag}/gen_best_rmse", float(metrics["rmse_j"][best_i, ji]), gi
        )
    best_decoded = decode_baseline(early["best_x"], groups, joints)
    history.append(
      {
        "gen": gi,
        "best_cost": best_cost,
        "running_best_cost": early["best_cost"],
        "stale": early["stale"],
        "mean_cost": float(np.mean(costs)),
        "best": best_decoded,
      }
    )
    print(
      f"[fullbody] gen={gi:03d} pop={len(costs)}  "
      f"best_cost={best_cost:.4f} running_best={early['best_cost']:.4f}  "
      f"stale={early['stale']} mean={float(np.mean(costs)):.4f}",
      flush=True,
    )
    if (
      video_env is not None
      and video_ctrl_map is not None
      and video_interval > 0
      and gi % video_interval == 0
    ):
      try:
        vpath = record_test_video(
          video_env,
          joints,
          video_ctrl_map,
          best_decoded,
          log_dir / "videos",
          args.video_duration,
          args.period_slow,
          args.period_fast,
          writer=writer,
          gen=gi,
          name_prefix="walk_track",
          grid_cols=args.video_tile_cols,
          ref_amp=ref_amp,
        )
        videos_recorded.append(str(vpath))
      except Exception as e:
        print(f"[pd-tune] interval video failed gen={gi}: {e}", flush=True)
    # checkpoint
    (log_dir / "best_pd.json").write_text(
      json.dumps(
        {
          "method": "differential_evolution_fullbody_symmetric",
          "partial": True,
          "gen": gi,
          "best_cost": early["best_cost"],
          "gen_best_cost": best_cost,
          "stale": early["stale"],
          "symmetry": {g["stem"]: g["members"] for g in groups},
          "final": best_decoded,
          "videos": videos_recorded,
          "history_tail": history[-5:],
        },
        indent=2,
      )
    )
    gen["i"] += 1
    writer.flush()
    return costs

  def early_stop_callback(_xk, convergence: float) -> bool:
    writer.add_scalar("fullbody/de_convergence", float(convergence), gen["i"])
    if (
      args.early_stop_cost is not None
      and early["best_cost"] <= args.early_stop_cost
    ):
      early["reason"] = (
        f"cost_target running_best={early['best_cost']:.4f} "
        f"<= {args.early_stop_cost}"
      )
      print(f"[pd-tune] early stop: {early['reason']}", flush=True)
      return True
    if (
      args.early_stop_patience > 0
      and early["stale"] >= args.early_stop_patience
    ):
      early["reason"] = (
        f"plateau stale={early['stale']} "
        f"(delta={args.early_stop_delta}) running_best={early['best_cost']:.4f}"
      )
      print(f"[pd-tune] early stop: {early['reason']}", flush=True)
      return True
    return False

  result = differential_evolution(
    objective,
    bounds=bounds,
    strategy="best1bin",
    maxiter=args.maxiter,
    popsize=popsize,
    mutation=(0.5, 1.0),
    recombination=0.7,
    tol=1e-3,
    atol=1e-4,
    polish=False,
    seed=args.seed,
    vectorized=True,
    updating="deferred",
    workers=1,
    init="latinhypercube",
    x0=x0,
    callback=early_stop_callback,
  )

  # Prefer tracked running-best (noise makes last-gen DE result unreliable).
  best_x = np.asarray(early["best_x"], dtype=np.float64)
  best_cost = float(early["best_cost"])
  final = decode_baseline(best_x, groups, joints)

  # Final group + joint TB summary + wn/zeta
  for g in groups:
    name0 = g["members"][0]
    kp, kd = final[name0]
    arm = g["armature"]
    wn = math.sqrt(kp / arm) / (2 * math.pi)
    zeta = kd / (2 * math.sqrt(kp * arm))
    writer.add_scalar(f"group/{g['tag']}/best_kp", kp, 0)
    writer.add_scalar(f"group/{g['tag']}/best_kd", kd, 0)
    writer.add_scalar(f"group/{g['tag']}/best_wn_hz", wn, 0)
    writer.add_scalar(f"group/{g['tag']}/best_zeta", zeta, 0)
    for name in g["members"]:
      tag = _tb_tag(name)
      writer.add_scalar(f"joint/{tag}/best_kp", kp, 0)
      writer.add_scalar(f"joint/{tag}/best_kd", kd, 0)
      writer.add_scalar(f"joint/{tag}/best_wn_hz", wn, 0)
      writer.add_scalar(f"joint/{tag}/best_zeta", zeta, 0)
  writer.add_scalar("fullbody/best_cost", best_cost, 0)
  writer.add_scalar("fullbody/nfev", int(result.nfev), 0)
  writer.add_scalar("fullbody/nit", int(result.nit), 0)

  env.close()
  if video_env is not None:
    video_env.close()

  out = {
    "method": "differential_evolution_fullbody_symmetric_logpd",
    "num_envs": args.num_envs,
    "n_params": n_params,
    "n_groups": n_g,
    "n_joints": n_j,
    "symmetry": {g["stem"]: g["members"] for g in groups},
    "pd_parametrization": "log_per_lr_group_seed_mult_box",
    "kp_mult": list(kp_mult),
    "kd_mult": list(kd_mult),
    "ref_amp": ref_amp,
    "group_pd_bounds": {
      g["stem"]: {
        "kp": [round(g["kp0"] * kp_mult[0], 3), round(g["kp0"] * kp_mult[1], 3)],
        "kd": [round(g["kd0"] * kd_mult[0], 3), round(g["kd0"] * kd_mult[1], 3)],
      }
      for g in groups
    },
    "period_slow": args.period_slow,
    "period_fast": args.period_fast,
    "popsize": popsize,
    "maxiter": args.maxiter,
    "duration": args.duration,
    "noise_std": args.noise_std,
    "video_interval": video_interval,
    "early_stop_patience": args.early_stop_patience,
    "early_stop_delta": args.early_stop_delta,
    "early_stop_cost": args.early_stop_cost,
    "early_stop_reason": early["reason"],
    "best_cost": best_cost,
    "nfev": int(result.nfev),
    "nit": int(result.nit),
    "success": bool(result.success),
    "message": str(result.message),
    "final": final,
    "history": history,
    "videos": videos_recorded,
  }
  out_path = log_dir / "best_pd.json"
  out_path.write_text(json.dumps(out, indent=2))
  writer.add_text("final_baseline", json.dumps(final, indent=2), 0)
  writer.close()
  print(f"[pd-tune] done → {out_path}", flush=True)
  print("FINAL", json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
  main()
