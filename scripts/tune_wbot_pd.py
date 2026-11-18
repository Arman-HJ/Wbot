#!/usr/bin/env python3
"""Optimize WBOT joint PD (kp, kd) via step + sine tracking in MuJoCo.

Uses the measured armature from ``wbot_20dof_constants``. Floating base is pinned
each step; contacts disabled. One representative joint per actuator group is
optimized while others are held at the stand home pose.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from src.assets.robots.wbot.wbot_20dof_constants import (
  ARMATURE_E2806,
  ARMATURE_E4310,
  ARMATURE_E4315,
  ARMATURE_EL05,
  ARMATURE_RS00,
  WBOT_20DOF_HOME_STAND,
  get_spec,
)

DT = 0.005  # match env mujoco timestep
CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
SIM_STEPS_PER_CTRL = int(round(CTRL_DT / DT))


@dataclass(frozen=True)
class JointGroup:
  name: str
  joint: str  # representative joint
  armature: float
  effort: float
  kp_bounds: tuple[float, float]
  kd_bounds: tuple[float, float]


GROUPS = (
  JointGroup("RS00", "left_shoulder_pitch_joint", ARMATURE_RS00, 10.0, (2.0, 80.0), (0.05, 8.0)),
  JointGroup("EL05", "left_shoulder_yaw_joint", ARMATURE_EL05, 4.0, (1.0, 60.0), (0.05, 6.0)),
  JointGroup("E2806", "left_ankle_roll_joint", ARMATURE_E2806, 8.0, (3.0, 100.0), (0.1, 10.0)),
  JointGroup("E4310", "left_hip_yaw_joint", ARMATURE_E4310, 28.0, (5.0, 160.0), (0.2, 16.0)),
  JointGroup("E4315", "left_hip_pitch_joint", ARMATURE_E4315, 65.0, (10.0, 220.0), (0.3, 20.0)),
)


def _home_joint_map() -> dict[str, float]:
  raw = dict(WBOT_20DOF_HOME_STAND.joint_pos or {})
  # Expand regex-like keys used in EntityCfg into concrete names when needed.
  # HOME_STAND already uses concrete left/right names.
  return {k: float(v) for k, v in raw.items() if "*" not in k}


def build_model() -> tuple[mujoco.MjModel, dict[str, int], dict[str, int], np.ndarray]:
  model = get_spec().compile()
  model.opt.timestep = DT
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

  qadr: dict[str, int] = {}
  dadr: dict[str, int] = {}
  for i in range(model.njnt):
    j = model.joint(i)
    if j.type != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    qadr[j.name] = int(j.qposadr[0])
    dadr[j.name] = int(j.dofadr[0])

  home = np.array(model.qpos0, dtype=np.float64, copy=True)
  # Keep compiled keyframe height; overlay stand joint angles.
  for name, ang in _home_joint_map().items():
    if name in qadr:
      home[qadr[name]] = ang

  # Apply measured armature on all hinge dofs (default, overwritten per-group below).
  armature_by_name = {
    "shoulder_pitch": ARMATURE_RS00,
    "shoulder_roll": ARMATURE_RS00,
    "shoulder_yaw": ARMATURE_EL05,
    "elbow": ARMATURE_EL05,
    "ankle_roll": ARMATURE_E2806,
    "hip_yaw": ARMATURE_E4310,
    "ankle_pitch": ARMATURE_E4310,
    "hip_pitch": ARMATURE_E4315,
    "hip_roll": ARMATURE_E4315,
    "knee": ARMATURE_E4315,
  }
  for name, di in dadr.items():
    for key, arm in armature_by_name.items():
      if key in name:
        model.dof_armature[di] = arm
        break
  return model, qadr, dadr, home


def simulate(
  model: mujoco.MjModel,
  qadr: dict[str, int],
  dadr: dict[str, int],
  home: np.ndarray,
  joint: str,
  kp: float,
  kd: float,
  effort: float,
  mode: str,
  duration: float,
  amp: float,
  freq_hz: float = 1.0,
  hold_kp: float = 80.0,
  hold_kd: float = 4.0,
) -> dict[str, float]:
  data = mujoco.MjData(model)
  data.qpos[:] = home
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  qi, di = qadr[joint], dadr[joint]
  q0 = float(home[qi])
  n_ctrl = int(round(duration / CTRL_DT))
  err2 = 0.0
  overshoot = 0.0
  tau_abs = 0.0
  n = 0
  peak = q0
  settled_band = 0.02 * abs(amp) + 1e-4
  last_enter = None
  osc_penalty = 0.0
  prev_err = 0.0

  base_qpos = home[:7].copy()

  for k in range(n_ctrl):
    t = k * CTRL_DT
    if mode == "step":
      q_des = q0 + (amp if t >= 0.2 else 0.0)
    elif mode == "sine":
      q_des = q0 + amp * math.sin(2.0 * math.pi * freq_hz * t)
    else:
      raise ValueError(mode)

    # Pin floating base.
    data.qpos[:7] = base_qpos
    data.qvel[:6] = 0.0

    # PD on all hinges: hold others, track target on active joint.
    for name, qj in qadr.items():
      dj = dadr[name]
      q = float(data.qpos[qj])
      dq = float(data.qvel[dj])
      if name == joint:
        tau = kp * (q_des - q) - kd * dq
        tau = float(np.clip(tau, -effort, effort))
      else:
        tau = hold_kp * (float(home[qj]) - q) - hold_kd * dq
        tau = float(np.clip(tau, -80.0, 80.0))
      data.qfrc_applied[dj] = tau

    for _ in range(SIM_STEPS_PER_CTRL):
      data.qpos[:7] = base_qpos
      data.qvel[:6] = 0.0
      mujoco.mj_step(model, data)

    q = float(data.qpos[qi])
    e = q_des - q
    err2 += e * e
    tau_abs += abs(float(data.qfrc_applied[di]))
    n += 1
    if mode == "step" and t >= 0.2:
      if amp > 0:
        peak = max(peak, q)
        overshoot = max(overshoot, (peak - (q0 + amp)) / max(abs(amp), 1e-6))
      else:
        peak = min(peak, q)
        overshoot = max(overshoot, ((q0 + amp) - peak) / max(abs(amp), 1e-6))
      if abs(e) < settled_band:
        if last_enter is None:
          last_enter = t
      else:
        last_enter = None
      # chatter: error sign flips with large magnitude
      if e * prev_err < 0 and abs(e) > 0.5 * abs(amp):
        osc_penalty += 1.0
      prev_err = e

  rmse = math.sqrt(err2 / max(n, 1))
  tau_mean = tau_abs / max(n, 1)
  settle = 0.0
  if mode == "step":
    settle = 0.0 if last_enter is not None else 1.0
  return {
    "rmse": rmse,
    "overshoot": max(overshoot, 0.0),
    "tau_mean": tau_mean,
    "settle_fail": settle,
    "osc": osc_penalty / max(n, 1),
  }


def cost_for_group(
  model: mujoco.MjModel,
  qadr: dict[str, int],
  dadr: dict[str, int],
  home: np.ndarray,
  group: JointGroup,
  kp: float,
  kd: float,
) -> float:
  # Keep zeta-ish region soft-preferred: kd ~ 2*zeta*sqrt(kp*I), zeta in [0.7, 2.0]
  i = group.armature
  kd_lo = 2.0 * 0.6 * math.sqrt(max(kp * i, 1e-12))
  kd_hi = 2.0 * 2.2 * math.sqrt(max(kp * i, 1e-12))
  zeta_pen = 0.0
  if kd < kd_lo:
    zeta_pen += (kd_lo - kd) / max(kd_lo, 1e-6)
  if kd > kd_hi:
    zeta_pen += (kd - kd_hi) / max(kd_hi, 1e-6)

  step = simulate(
    model, qadr, dadr, home, group.joint, kp, kd, group.effort,
    mode="step", duration=1.5, amp=0.20,
  )
  step_n = simulate(
    model, qadr, dadr, home, group.joint, kp, kd, group.effort,
    mode="step", duration=1.5, amp=-0.20,
  )
  sine1 = simulate(
    model, qadr, dadr, home, group.joint, kp, kd, group.effort,
    mode="sine", duration=2.0, amp=0.15, freq_hz=1.0,
  )
  sine2 = simulate(
    model, qadr, dadr, home, group.joint, kp, kd, group.effort,
    mode="sine", duration=2.0, amp=0.12, freq_hz=2.0,
  )

  # Weighted multi-objective → scalar.
  c = 0.0
  c += 4.0 * (step["rmse"] + step_n["rmse"])
  c += 1.5 * (step["overshoot"] + step_n["overshoot"])
  c += 2.0 * (step["settle_fail"] + step_n["settle_fail"])
  c += 3.0 * (step["osc"] + step_n["osc"])
  c += 5.0 * sine1["rmse"] + 6.0 * sine2["rmse"]
  c += 0.002 * (step["tau_mean"] + sine1["tau_mean"] + sine2["tau_mean"])
  c += 0.5 * zeta_pen
  # Mild preference against absurdly high kp (noise sensitivity on real robot).
  c += 0.0005 * max(kp - 120.0, 0.0)
  return float(c)


def optimize_group(
  model: mujoco.MjModel,
  qadr: dict[str, int],
  dadr: dict[str, int],
  home: np.ndarray,
  group: JointGroup,
  maxiter: int,
  seed: int,
) -> dict:
  def obj(x: np.ndarray) -> float:
    return cost_for_group(model, qadr, dadr, home, group, float(x[0]), float(x[1]))

  result = differential_evolution(
    obj,
    bounds=[group.kp_bounds, group.kd_bounds],
    maxiter=maxiter,
    popsize=12,
    mutation=(0.5, 1.2),
    recombination=0.7,
    seed=seed,
    polish=True,
    workers=1,
    updating="immediate",
    tol=1e-3,
  )
  kp, kd = float(result.x[0]), float(result.x[1])
  metrics = {
    "step": simulate(
      model, qadr, dadr, home, group.joint, kp, kd, group.effort,
      mode="step", duration=1.5, amp=0.20,
    ),
    "sine_1hz": simulate(
      model, qadr, dadr, home, group.joint, kp, kd, group.effort,
      mode="sine", duration=2.0, amp=0.15, freq_hz=1.0,
    ),
    "sine_2hz": simulate(
      model, qadr, dadr, home, group.joint, kp, kd, group.effort,
      mode="sine", duration=2.0, amp=0.12, freq_hz=2.0,
    ),
  }
  wn = math.sqrt(kp / group.armature) / (2.0 * math.pi)
  zeta = kd / (2.0 * math.sqrt(kp * group.armature))
  return {
    "group": group.name,
    "joint": group.joint,
    "kp": round(kp, 3),
    "kd": round(kd, 3),
    "armature": group.armature,
    "effort": group.effort,
    "wn_hz": round(wn, 2),
    "zeta": round(zeta, 3),
    "cost": float(result.fun),
    "success": bool(result.success),
    "metrics": metrics,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--maxiter", type=int, default=25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("logs/pd_tune/wbot_20dof_pd_opt.json"),
  )
  args = parser.parse_args()

  model, qadr, dadr, home = build_model()
  results = []
  for g in GROUPS:
    print(f"[tune] {g.name} ({g.joint}) armature={g.armature:.6g} ...", flush=True)
    r = optimize_group(model, qadr, dadr, home, g, maxiter=args.maxiter, seed=args.seed)
    results.append(r)
    print(
      f"  -> kp={r['kp']:.3f} kd={r['kd']:.3f}  wn={r['wn_hz']}Hz  "
      f"zeta={r['zeta']}  cost={r['cost']:.4f}  "
      f"step_rmse={r['metrics']['step']['rmse']:.4f}  "
      f"sine2_rmse={r['metrics']['sine_2hz']['rmse']:.4f}",
      flush=True,
    )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  payload = {"dt": DT, "ctrl_hz": CTRL_HZ, "groups": results}
  args.out.write_text(json.dumps(payload, indent=2))
  print(f"[tune] wrote {args.out}", flush=True)


if __name__ == "__main__":
  main()
