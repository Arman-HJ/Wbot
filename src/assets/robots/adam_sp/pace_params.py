"""PACE-identified leg actuator parameters loaded from YAML (no hardcoded fit values)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg

_ADAM_SP_PKG = Path(__file__).resolve().parent
DEFAULT_ADAM_SP_LEGS_PACE_YAML = _ADAM_SP_PKG / "params" / "adam_sp_legs_real_pace.yaml"
DEFAULT_PND_LEGS_PACE_YAML = DEFAULT_ADAM_SP_LEGS_PACE_YAML

# Effort / Kp / Kd for 12 leg joints (pnd_rl_gym adam_lite_12dof_config.control).
PND_LEG_PD: dict[str, tuple[float, float, float]] = {
  "hipPitch_Left": (230.0, 305.0, 6.1),
  "hipPitch_Right": (230.0, 305.0, 6.1),
  "hipRoll_Left": (160.0, 700.0, 30.0),
  "hipRoll_Right": (160.0, 700.0, 30.0),
  "hipYaw_Left": (105.0, 405.0, 6.1),
  "hipYaw_Right": (105.0, 405.0, 6.1),
  "kneePitch_Left": (230.0, 305.0, 6.1),
  "kneePitch_Right": (230.0, 305.0, 6.1),
  "anklePitch_Left": (40.0, 25.0, 2.55),
  "anklePitch_Right": (40.0, 25.0, 2.55),
  "ankleRoll_Left": (12.0, 1.0e-3, 0.35),
  "ankleRoll_Right": (12.0, 1.0e-3, 0.35),
}


@dataclass(frozen=True)
class IdentifiedJointParams:
  armature: float
  viscous_friction: float
  friction: float
  encoder_bias_rad: float


@dataclass
class IdentifiedParamsDoc:
  robot: str
  joint_order: list[str]
  delay_sim_steps: float
  joints: dict[str, IdentifiedJointParams]
  fit_meta: dict[str, Any] = field(default_factory=dict)

  def delay_lag_steps(self) -> int:
    return max(0, int(round(self.delay_sim_steps)))

  def encoder_bias_by_joint(self) -> dict[str, float]:
    return {name: self.joints[name].encoder_bias_rad for name in self.joint_order}


def resolve_pnd_legs_pace_yaml(path: str | Path | None = None) -> Path:
  if path is not None:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
      raise FileNotFoundError(f"PACE params YAML not found: {p}")
    return p
  env = os.environ.get("PND_LEGS_PACE_YAML")
  if env:
    p = Path(env).expanduser().resolve()
    if not p.is_file():
      raise FileNotFoundError(f"PND_LEGS_PACE_YAML not found: {p}")
    return p
  if not DEFAULT_PND_LEGS_PACE_YAML.is_file():
    raise FileNotFoundError(
      f"Default PACE params missing: {DEFAULT_PND_LEGS_PACE_YAML}. "
      "Set PND_LEGS_PACE_YAML or pass pace_params_yaml explicitly."
    )
  return DEFAULT_PND_LEGS_PACE_YAML


def load_pnd_legs_pace_params(path: str | Path | None = None) -> IdentifiedParamsDoc:
  yaml_path = resolve_pnd_legs_pace_yaml(path)
  with open(yaml_path, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  if not isinstance(raw, dict):
    raise ValueError(f"PACE YAML must be a mapping: {yaml_path}")

  joint_order = raw.get("joint_order")
  if not isinstance(joint_order, list) or not joint_order:
    raise ValueError(f"joint_order missing in {yaml_path}")

  joints_raw = raw.get("joints")
  if not isinstance(joints_raw, dict):
    raise ValueError(f"joints missing in {yaml_path}")

  joints: dict[str, IdentifiedJointParams] = {}
  for name in joint_order:
    name = str(name)
    entry = joints_raw.get(name)
    if not isinstance(entry, dict):
      raise ValueError(f"joint {name!r} missing in {yaml_path}")
    joints[name] = IdentifiedJointParams(
      armature=float(entry["armature"]),
      viscous_friction=float(entry["viscous_friction"]),
      friction=float(entry["friction"]),
      encoder_bias_rad=float(entry["encoder_bias_rad"]),
    )

  fit_meta = raw.get("fit_meta")
  return IdentifiedParamsDoc(
    robot=str(raw.get("robot", "unknown")),
    joint_order=[str(x) for x in joint_order],
    delay_sim_steps=float(raw.get("delay_sim_steps", 0.0)),
    joints=joints,
    fit_meta=dict(fit_meta) if isinstance(fit_meta, dict) else {},
  )


def build_legs_only_articulation_from_pace(
  pace_params_yaml: str | Path | None = None,
) -> EntityArticulationInfoCfg:
  """One BuiltinPositionActuatorCfg per leg joint with PACE dynamics from YAML."""
  doc = load_pnd_legs_pace_params(pace_params_yaml)
  delay_lag = doc.delay_lag_steps()
  actuators: list[BuiltinPositionActuatorCfg] = []
  for name in doc.joint_order:
    if name not in PND_LEG_PD:
      raise KeyError(f"No PD table entry for leg joint {name!r}")
    effort, kp, kd = PND_LEG_PD[name]
    pace = doc.joints[name]
    actuators.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=kp,
        damping=kd,
        effort_limit=effort,
        armature=pace.armature,
        frictionloss=pace.friction,
        viscous_damping=pace.viscous_friction,
        delay_min_lag=delay_lag,
        delay_max_lag=delay_lag,
      )
    )
  return EntityArticulationInfoCfg(
    actuators=tuple(actuators),
    soft_joint_pos_limit_factor=0.9,
  )
