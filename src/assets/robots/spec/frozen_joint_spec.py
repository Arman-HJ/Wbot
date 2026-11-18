"""Bake matched joints into rigid body poses (shared by Z1 legs+waist spec)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from mjlab.entity import EntityCfg

_LEGS_SPEC_XML_LOGGED: set[str] = set()
_MASS_LIGHTEN_LOGGED = False
_MASS_SUMMARY_LOGGED = False

# Optional mass/COM tweaks when baking upper-body joints (defaults: no change).
_BAKED_SUBTREE_MASS_SCALE = 1.0
_TORSO_MASS_SCALE = 1.0
_TORSO_IPOS_OFFSET = (0.0, 0.0, 0.0)
_PELVIS_BODY = "pelvis"
_TORSO_BODY = "torso"
_TORSO_IMU_SITE = "torso_imu"
_TORSO_UPVECTOR_SENSOR = "torso_upvector"


def _strip_xml_actuators(spec: mujoco.MjSpec) -> None:
  """Remove MJCF actuators so mjlab can inject BuiltinPositionActuator."""
  for act in list(spec.actuators):
    spec.delete(act)

# Legs + floating base only; everything else is baked rigid.
_LEG_JOINT_RE = re.compile(
  r"^(hipPitch_|hipRoll_|hipYaw_|kneePitch_|anklePitch_|ankleRoll_).*$"
)

# Waist only: baked at LOCOMOTION_KEYFRAME, arms remain articulated.
_WAIST_JOINT_REMOVE_RE = re.compile(r"^(waistRoll|waistPitch|waistYaw)$")

# Legs-only: waist, arms, hands, neck baked at LOCOMOTION_KEYFRAME pose.
_UPPER_BODY_JOINT_REMOVE_RE = re.compile(
  r"^(waistRoll|waistPitch|waistYaw|"
  r"shoulderPitch_.*|shoulderRoll_.*|shoulderYaw_.*|elbow_.*|wrist.*|"
  r"L_.*|R_.*|"
  r"neckYaw|neckPitch)$"
)

# Locomotion: bake finger + neck joints only; legs, waist, arms remain actuated.
_HANDS_NECK_REMOVE_RE = re.compile(r"^(L_.*|R_.*|neckYaw|neckPitch)$")

# Legs+waist: arms/hands/neck baked, keep waist joints controllable.
ARMS_HANDS_NECK_REMOVE_RE = re.compile(
  r"^(shoulderPitch_.*|shoulderRoll_.*|shoulderYaw_.*|elbow_.*|wrist.*|"
  r"L_.*|R_.*|"
  r"neckYaw|neckPitch)$"
)

# Deploy arm-sway test: only shoulder pitch free; other arm/hand/neck joints baked.
_ARMS_SWAY_KEEP_SHOULDER_PITCH_RE = re.compile(
  r"^(shoulderRoll_.*|shoulderYaw_.*|elbow_.*|wrist.*|"
  r"L_.*|R_.*|"
  r"neckYaw|neckPitch)$"
)


def _is_retained_joint(name: str) -> bool:
  if name == "floating_base":
    return True
  return _LEG_JOINT_RE.match(name) is not None


def _joints_to_remove(spec: mujoco.MjSpec, remove_re: re.Pattern[str]) -> list[str]:
  return sorted(j.name for j in spec.joints if j.name and remove_re.match(j.name))


def _mat_to_quat(xmat: np.ndarray) -> np.ndarray:
  quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quat, xmat)
  return quat


def _body_rel_pose(
  model: mujoco.MjModel, data: mujoco.MjData, parent_id: int, child_id: int
) -> tuple[np.ndarray, np.ndarray]:
  parent_pos = data.xpos[parent_id]
  child_pos = data.xpos[child_id]
  parent_mat = data.xmat[parent_id].reshape(3, 3)
  rel_pos = parent_mat.T @ (child_pos - parent_pos)

  parent_quat = _mat_to_quat(data.xmat[parent_id])
  child_quat = _mat_to_quat(data.xmat[child_id])
  parent_quat_conj = np.zeros(4, dtype=np.float64)
  mujoco.mju_negQuat(parent_quat_conj, parent_quat)
  rel_quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mulQuat(rel_quat, parent_quat_conj, child_quat)
  return rel_pos, rel_quat


def _joint_body_depth(model: mujoco.MjModel, joint_name: str) -> int:
  body_id = int(model.joint(joint_name).bodyid[0])
  depth = 0
  while True:
    parent_id = int(model.body(body_id).parentid[0])
    if parent_id <= 0:
      break
    depth += 1
    body_id = parent_id
  return depth


def _find_spec_body_for_joint(spec: mujoco.MjSpec, joint_name: str) -> mujoco.MjsBody:
  for body in spec.bodies:
    for joint in body.joints:
      if joint.name == joint_name:
        return body
  raise KeyError(f"Joint {joint_name!r} not found on any body in spec")


def _apply_keyframe_qpos(
  model: mujoco.MjModel, data: mujoco.MjData, joint_pos: dict[str, float]
) -> None:
  for name, value in joint_pos.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      continue
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr] = float(value)
  for name in ("neckYaw", "neckPitch"):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
      data.qpos[int(model.jnt_qposadr[jid])] = 0.0


def _collect_baked_poses(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_names: Iterable[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  baked: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  for jname in joint_names:
    child_id = int(model.joint(jname).bodyid[0])
    parent_id = int(model.body(child_id).parentid[0])
    if parent_id <= 0:
      continue
    baked[jname] = _body_rel_pose(model, data, parent_id, child_id)
  return baked


def _delete_sensors_for_joints(spec: mujoco.MjSpec, joint_names: set[str]) -> None:
  to_delete: list[mujoco.MjsSensor] = []
  for sensor in spec.sensors:
    jname = getattr(sensor, "objname", None) or getattr(sensor, "joint", None)
    if jname and jname in joint_names:
      to_delete.append(sensor)
  for sensor in to_delete:
    spec.delete(sensor)


def _delete_equalities_for_joints(spec: mujoco.MjSpec, joint_names: set[str]) -> None:
  to_delete = []
  for eq in spec.equalities:
    jname = getattr(eq, "name1", None) or getattr(eq, "joint1", None)
    if jname and jname in joint_names:
      to_delete.append(eq)
  for eq in to_delete:
    spec.delete(eq)


def _freeze_joint_in_spec(
  spec: mujoco.MjSpec,
  joint_name: str,
  rel_pos: np.ndarray,
  rel_quat: np.ndarray,
) -> None:
  body = _find_spec_body_for_joint(spec, joint_name)
  if not body.joints:
    return
  joint = body.joints[0]
  if joint.name != joint_name:
    for candidate in body.joints:
      if candidate.name == joint_name:
        joint = candidate
        break
  spec.delete(joint)
  body.pos[:] = rel_pos.tolist()
  body.quat[:] = rel_quat.tolist()


def _ensure_foot_sites(spec: mujoco.MjSpec) -> None:
  """Ensure toe-attached sites exist for task sensors/rewards."""
  existing = {s.name for s in spec.sites if s.name}
  targets = (("left_foot", "toeLeft"), ("right_foot", "toeRight"))
  for site_name, body_name in targets:
    if site_name in existing:
      continue
    body = next((b for b in spec.bodies if b.name == body_name), None)
    if body is None:
      continue
    body.add_site(name=site_name, pos=[0.0, 0.0, 0.0])


def _ensure_torso_gravity_sensor(spec: mujoco.MjSpec) -> None:
  """Torso-frame projected gravity (FK from pelvis); pelvis IMU kept for ang vel only."""
  site_names = {s.name for s in spec.sites if s.name}
  torso = next((b for b in spec.bodies if b.name == _TORSO_BODY), None)
  if torso is None:
    return
  if _TORSO_IMU_SITE not in site_names:
    torso.add_site(name=_TORSO_IMU_SITE, pos=[0.0, 0.0, 0.0])
  sensor_names = {s.name for s in spec.sensors if s.name}
  if _TORSO_UPVECTOR_SENSOR in sensor_names:
    return
  spec.add_sensor(
    name=_TORSO_UPVECTOR_SENSOR,
    type=int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS),
    objtype=int(mujoco.mjtObj.mjOBJ_BODY),
    objname="world",
    reftype=int(mujoco.mjtObj.mjOBJ_SITE),
    refname=_TORSO_IMU_SITE,
  )


def _ensure_geom_names(spec: mujoco.MjSpec) -> None:
  """Assign names to unnamed geoms to support regex-based scene selection."""
  used = {g.name for g in spec.geoms if g.name}
  for i, geom in enumerate(spec.geoms):
    if geom.name:
      continue
    mesh_name = getattr(geom, "meshname", None) or getattr(geom, "mesh", None)
    base = mesh_name if isinstance(mesh_name, str) and mesh_name else "geom"
    candidate = f"{base}_collision"
    if candidate in used:
      candidate = f"{base}_{i}_collision"
    while candidate in used:
      candidate = f"{candidate}_{i}"
    geom.name = candidate
    used.add(candidate)


def _collect_subtree_body_names(body: mujoco.MjsBody) -> set[str]:
  names: set[str] = set()
  if body.name:
    names.add(body.name)
  for child in body.bodies:
    names |= _collect_subtree_body_names(child)
  return names


def _collect_frozen_subtree_body_names(
  spec: mujoco.MjSpec, remove_re: re.Pattern[str]
) -> set[str]:
  """Bodies whose joints will be baked/removed (arms/hands/neck or full upper body)."""
  names: set[str] = set()
  for body in spec.bodies:
    for joint in body.joints:
      if joint.name and remove_re.match(joint.name):
        names |= _collect_subtree_body_names(body)
  return names


def _body_masses_from_spec(spec: mujoco.MjSpec) -> dict[str, float]:
  """Compile once and read per-body mass (kg), excluding world body."""
  model = spec.compile()
  masses: dict[str, float] = {}
  for i in range(1, model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
    masses[name] = float(model.body_mass[i])
  return masses


def _mass_summary(
  masses: dict[str, float], frozen_body_names: set[str]
) -> dict[str, float]:
  total = sum(masses.values())
  pelvis = masses.get(_PELVIS_BODY, 0.0)
  torso = masses.get(_TORSO_BODY, 0.0)
  baked = sum(m for name, m in masses.items() if name in frozen_body_names)
  other = total - pelvis - torso - baked
  return {
    "total": total,
    "pelvis": pelvis,
    "torso": torso,
    "baked_upper": baked,
    "other": other,
  }


def _format_mass_summary_line(tag: str, summary: dict[str, float]) -> str:
  return (
    f"[INFO] PND legs model mass ({tag}): "
    f"total={summary['total']:.3f} kg | "
    f"pelvis={summary['pelvis']:.3f} kg | "
    f"torso={summary['torso']:.3f} kg | "
    f"baked_upper={summary['baked_upper']:.3f} kg | "
    f"other={summary['other']:.3f} kg"
  )


def _log_mass_summary_once(
  before: dict[str, float],
  after: dict[str, float],
  frozen_body_names: set[str],
  *,
  spec: mujoco.MjSpec | None = None,
  locomotion_keyframe: EntityCfg.InitialStateCfg | None = None,
) -> None:
  global _MASS_SUMMARY_LOGGED
  if _MASS_SUMMARY_LOGGED:
    return
  _MASS_SUMMARY_LOGGED = True

  before_s = _mass_summary(before, frozen_body_names)
  after_s = _mass_summary(after, frozen_body_names)
  print(_format_mass_summary_line("before tweak", before_s), flush=True)
  if before_s != after_s:
    print(_format_mass_summary_line("after tweak", after_s), flush=True)
    delta = after_s["total"] - before_s["total"]
    pct = 100.0 * delta / before_s["total"] if before_s["total"] > 0 else 0.0
    if delta >= 0:
      print(
        f"[INFO] PND legs mass delta: +{delta:.3f} kg added ({pct:.1f}%)",
        flush=True,
      )
    else:
      print(
        f"[INFO] PND legs mass delta: {delta:.3f} kg removed ({pct:.1f}%)",
        flush=True,
      )

  heavy = sorted(after.items(), key=lambda kv: kv[1], reverse=True)
  heavy = [(n, m) for n, m in heavy if m >= 0.5][:12]
  if heavy:
    parts = ", ".join(f"{name}={mass:.3f}" for name, mass in heavy)
    print(f"[INFO] PND legs heaviest bodies (>=0.5 kg): {parts}", flush=True)

  if spec is not None and locomotion_keyframe is not None:
    _print_standing_com_snapshot(spec, locomotion_keyframe)


def _print_standing_com_snapshot(
  spec: mujoco.MjSpec, locomotion_keyframe: EntityCfg.InitialStateCfg
) -> None:
  """Print sagittal COM offset at default standing pose (for forward/back tuning)."""
  model = spec.compile()
  data = mujoco.MjData(model)
  assert locomotion_keyframe.joint_pos is not None
  _apply_keyframe_qpos(model, data, locomotion_keyframe.joint_pos)
  mujoco.mj_forward(model, data)

  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PELVIS_BODY)
  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _TORSO_BODY)
  if pelvis_id < 0:
    return

  pelvis_com = np.asarray(data.subtree_com[pelvis_id], dtype=np.float64)
  pelvis_pos = np.asarray(data.xpos[pelvis_id], dtype=np.float64)
  sagittal = float(pelvis_com[0] - pelvis_pos[0])
  msg = (
    "[INFO] PND standing COM (sagittal): "
    f"pelvis_subtree_com_x={pelvis_com[0]:.4f} m, "
    f"pelvis_body_x={pelvis_pos[0]:.4f} m, "
    f"com_ahead_of_pelvis={sagittal:+.4f} m"
  )
  if torso_id >= 0:
    torso_com = np.asarray(data.subtree_com[torso_id], dtype=np.float64)
    msg += (
      f", torso_subtree_com_x={torso_com[0]:.4f} m, "
      f"torso_ahead_of_pelvis={float(torso_com[0] - pelvis_pos[0]):+.4f} m"
    )
  offset = _TORSO_IPOS_OFFSET
  if any(abs(v) > 1e-9 for v in offset):
    msg += f", torso_ipos_offset={tuple(float(v) for v in offset)}"
  print(msg, flush=True)


def _scale_spec_body_mass(body: mujoco.MjsBody, scale: float) -> None:
  if abs(scale - 1.0) < 1e-9:
    return
  mass = float(body.mass)
  if mass <= 0.0:
    return
  body.explicitinertial = 1
  body.mass = mass * scale
  body.inertia[:] = np.asarray(body.inertia, dtype=np.float64) * scale


def _apply_temporary_mass_scales(
  spec: mujoco.MjSpec, frozen_body_names: set[str]
) -> None:
  """Temporary: scale baked upper-body and torso mass."""
  global _MASS_LIGHTEN_LOGGED
  if (
    _BAKED_SUBTREE_MASS_SCALE == 1.0
    and _TORSO_MASS_SCALE == 1.0
  ):
    return

  scaled_any = False
  for body in spec.bodies:
    if not body.name:
      continue
    scale = 1.0
    if (
      body.name in frozen_body_names
      and _BAKED_SUBTREE_MASS_SCALE != 1.0
    ):
      scale = _BAKED_SUBTREE_MASS_SCALE
    elif (
      body.name == _TORSO_BODY
      and _TORSO_MASS_SCALE != 1.0
    ):
      scale = _TORSO_MASS_SCALE
    if scale != 1.0:
      _scale_spec_body_mass(body, scale)
      scaled_any = True

  if scaled_any and not _MASS_LIGHTEN_LOGGED:
    print(
      "[INFO] baked-joint mass scale: "
      f"baked subtree x{_BAKED_SUBTREE_MASS_SCALE}, "
      f"torso x{_TORSO_MASS_SCALE}",
      flush=True,
    )
    _MASS_LIGHTEN_LOGGED = True


def _apply_temporary_torso_com_offset(spec: mujoco.MjSpec) -> None:
  """Shift torso inertial origin without changing leg masses (forward/back balance)."""
  offset = np.asarray(_TORSO_IPOS_OFFSET, dtype=np.float64)
  if offset.shape != (3,) or float(np.linalg.norm(offset)) < 1e-9:
    return
  for body in spec.bodies:
    if body.name != _TORSO_BODY:
      continue
    body.explicitinertial = 1
    body.ipos[:] = np.asarray(body.ipos, dtype=np.float64) + offset
    print(
      "[INFO] PND temporary torso ipos offset (body frame): "
      f"({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f}) m",
      flush=True,
    )
    return


def build_spec_with_frozen_joints(
  remove_re: re.Pattern[str],
  *,
  source_xml: Path,
  locomotion_keyframe: EntityCfg.InitialStateCfg,
  spec_log_label: str = "frozen joints",
) -> mujoco.MjSpec:
  """Delete matched joints and bake child-body pose into parent (no mjEQ_JOINT)."""
  xml_key = str(source_xml.resolve())
  if xml_key not in _LEGS_SPEC_XML_LOGGED:
    print(f"[INFO] {spec_log_label} spec XML: {xml_key}", flush=True)
    _LEGS_SPEC_XML_LOGGED.add(xml_key)
  spec = mujoco.MjSpec.from_file(str(source_xml))
  _strip_xml_actuators(spec)
  _ensure_geom_names(spec)
  _ensure_foot_sites(spec)
  _ensure_torso_gravity_sensor(spec)
  frozen_body_names = _collect_frozen_subtree_body_names(spec, remove_re)
  joint_names = _joints_to_remove(spec, remove_re)
  if not joint_names:
    return spec

  model = spec.compile()
  data = mujoco.MjData(model)
  assert locomotion_keyframe.joint_pos is not None
  _apply_keyframe_qpos(model, data, locomotion_keyframe.joint_pos)
  mujoco.mj_forward(model, data)

  baked = _collect_baked_poses(model, data, joint_names)
  joint_names_sorted = sorted(
    joint_names, key=lambda n: _joint_body_depth(model, n), reverse=True
  )

  remove_set = set(joint_names)
  _delete_sensors_for_joints(spec, remove_set)
  _delete_equalities_for_joints(spec, remove_set)

  for jname in joint_names_sorted:
    rel_pos, rel_quat = baked[jname]
    _freeze_joint_in_spec(spec, jname, rel_pos, rel_quat)

  masses_before = _body_masses_from_spec(spec)
  _apply_temporary_mass_scales(spec, frozen_body_names)
  _apply_temporary_torso_com_offset(spec)
  masses_after = _body_masses_from_spec(spec)
  _log_mass_summary_once(
    masses_before, masses_after, frozen_body_names, spec=spec, locomotion_keyframe=locomotion_keyframe
  )
  return spec
