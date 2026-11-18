"""Print env observation/action scales and robot init vs runtime defaults (train/play)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from numbers import Number
from typing import Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor.builtin_sensor import ObjRef


def _resolve_gain(gain: Any, joint_name: str) -> float | None:
  if isinstance(gain, Number):
    return float(gain)
  if isinstance(gain, dict):
    if joint_name in gain and isinstance(gain[joint_name], Number):
      return float(gain[joint_name])
    for expr, value in gain.items():
      if isinstance(value, Number) and re.fullmatch(expr, joint_name):
        return float(value)
  return None


def collect_joint_pd_map(env_cfg: ManagerBasedRlEnvCfg, env: ManagerBasedRlEnv) -> dict[str, tuple[float | None, float | None]]:
  """Per-joint (Kp, Kd) resolved from actuator cfg, keyed by joint name."""
  robot = env.scene["robot"]
  robot_cfg = env_cfg.scene.entities.get("robot")
  articulation = getattr(robot_cfg, "articulation", None) if robot_cfg is not None else None
  actuators = getattr(articulation, "actuators", None)
  pd_map: dict[str, tuple[float | None, float | None]] = {}
  if actuators is None:
    return pd_map
  for act in actuators:
    target_expr = getattr(act, "target_names_expr", ())
    if not target_expr:
      continue
    _, matched_names = robot.find_joints(target_expr, preserve_order=True)
    for joint_name in matched_names:
      kp = _resolve_gain(getattr(act, "stiffness", None), joint_name)
      kd = _resolve_gain(getattr(act, "damping", None), joint_name)
      pd_map[joint_name] = (kp, kd)
  return pd_map


def collect_joint_frictionloss_map(
  env_cfg: ManagerBasedRlEnvCfg, env: ManagerBasedRlEnv
) -> dict[str, float | None]:
  """Per-joint dry friction (N·m) from actuator cfg ``frictionloss``."""
  robot = env.scene["robot"]
  robot_cfg = env_cfg.scene.entities.get("robot")
  articulation = getattr(robot_cfg, "articulation", None) if robot_cfg is not None else None
  actuators = getattr(articulation, "actuators", None)
  fl_map: dict[str, float | None] = {}
  if actuators is None:
    return fl_map
  for act in actuators:
    target_expr = getattr(act, "target_names_expr", ())
    if not target_expr:
      continue
    _, matched_names = robot.find_joints(target_expr, preserve_order=True)
    fl = getattr(act, "frictionloss", None)
    fl_val = float(fl) if isinstance(fl, Number) else None
    for joint_name in matched_names:
      fl_map[joint_name] = fl_val
  return fl_map


def collect_joint_effort_limit_map(
  env_cfg: ManagerBasedRlEnvCfg, env: ManagerBasedRlEnv
) -> dict[str, float | None]:
  """Per-joint max torque (Nm) from actuator cfg, falling back to MuJoCo forcerange."""
  robot = env.scene["robot"]
  robot_cfg = env_cfg.scene.entities.get("robot")
  articulation = getattr(robot_cfg, "articulation", None) if robot_cfg is not None else None
  actuators = getattr(articulation, "actuators", None)
  effort_map: dict[str, float | None] = {}
  if actuators is not None:
    for act in actuators:
      target_expr = getattr(act, "target_names_expr", ())
      if not target_expr:
        continue
      _, matched_names = robot.find_joints(target_expr, preserve_order=True)
      for joint_name in matched_names:
        effort = _resolve_gain(getattr(act, "effort_limit", None), joint_name)
        effort_map[joint_name] = effort

  # Fallback / fill gaps from compiled MuJoCo actuator forcerange.
  try:
    mj_model = env.sim.mj_model
  except Exception:
    return effort_map
  for act_id in range(int(mj_model.nu)):
    act_name = str(mj_model.actuator(act_id).name)
    # Names are often ``robot/<joint_name>``.
    joint_name = act_name.split("/")[-1] if "/" in act_name else act_name
    if effort_map.get(joint_name) is not None:
      continue
    fr = mj_model.actuator_forcerange[act_id]
    # Symmetric torque limit → take abs of high side (or max abs of both).
    limit = float(max(abs(fr[0]), abs(fr[1])))
    if limit > 0:
      effort_map[joint_name] = limit
  return effort_map


def _noise_to_yaml(noise: Any) -> Any:
  if noise is None:
    return None
  n_min = getattr(noise, "n_min", None)
  n_max = getattr(noise, "n_max", None)
  if n_min is not None and n_max is not None:
    return {"type": "uniform", "n_min": float(n_min), "n_max": float(n_max)}
  return repr(noise)


def _to_float_list(x: Any) -> list[float] | Any:
  if x is None:
    return None
  try:
    return [float(v) for v in x]
  except (TypeError, ValueError):
    return x


def _scale_to_yaml(scale: Any) -> Any:
  """Serialize ObservationTermCfg.scale for YAML (scalar / list; strip torch tensors)."""
  if scale is None:
    return None
  if isinstance(scale, torch.Tensor):
    if scale.numel() == 1:
      return float(scale.item())
    return [float(x) for x in scale.detach().cpu().flatten().tolist()]
  if isinstance(scale, (float, int)):
    return float(scale)
  if isinstance(scale, tuple):
    return [float(x) for x in scale]
  if isinstance(scale, list) and all(isinstance(x, (int, float)) for x in scale):
    return [float(x) for x in scale]
  return repr(scale)


def _params_for_yaml(params: dict[str, Any] | None) -> dict[str, Any]:
  """Keep only JSON/YAML-safe observation term params (skip SceneEntityCfg etc.)."""
  if not params:
    return {}
  out: dict[str, Any] = {}
  for k, v in params.items():
    if isinstance(v, (str, int, float, bool)) or v is None:
      out[k] = v
    elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
      out[k] = [float(x) for x in v]
  return out


def _height_scan_sensor_yaml(env_cfg: ManagerBasedRlEnvCfg, env: ManagerBasedRlEnv) -> dict[str, Any] | None:
  """Serialize terrain_scan RayCastSensorCfg for deploy-side height_scan parity."""
  scene = getattr(env_cfg, "scene", None)
  if scene is None:
    return None
  sensors = getattr(scene, "sensors", None) or ()
  robot = env.scene["robot"]
  root_body_name = str(robot.root_body.name)
  for s in sensors:
    if getattr(s, "name", None) != "terrain_scan":
      continue
    frame = getattr(s, "frame", None)
    attachment_body_name = root_body_name
    if isinstance(frame, ObjRef) and frame.type == "body":
      nm = (frame.name or "").strip()
      attachment_body_name = nm if nm else root_body_name
    pat = getattr(s, "pattern", None)
    size = getattr(pat, "size", (1.6, 1.0))
    res = float(getattr(pat, "resolution", 0.1))
    direction = getattr(pat, "direction", (0.0, 0.0, -1.0))
    return {
      "sensor_name": "terrain_scan",
      "attachment_body_name": attachment_body_name,
      "ray_alignment": getattr(s, "ray_alignment", "yaw"),
      "max_distance": float(getattr(s, "max_distance", 5.0)),
      "exclude_parent_body": bool(getattr(s, "exclude_parent_body", True)),
      "include_geom_groups": (
        [int(g) for g in getattr(s, "include_geom_groups")] if getattr(s, "include_geom_groups", None) is not None else None
      ),
      "pattern": {
        "type": "grid",
        "size": [float(size[0]), float(size[1])],
        "resolution": res,
        "direction": [float(direction[0]), float(direction[1]), float(direction[2])],
      },
    }
  return None


def _joint_pos_cfg_to_yaml(jp_cfg: Any) -> Any:
  """Serialize ``init_state.joint_pos`` for YAML (dict name→rad is common; also batched vectors)."""
  if jp_cfg is None:
    return None
  if isinstance(jp_cfg, dict):
    return {
      str(k): float(v) if isinstance(v, Number) else v for k, v in jp_cfg.items()
    }
  try:
    return [float(v) for v in jp_cfg[0]]
  except (TypeError, ValueError, KeyError, IndexError):
    pass
  try:
    return [float(v) for v in jp_cfg]
  except (TypeError, ValueError):
    return repr(jp_cfg)


def _is_regex_pattern(s: str) -> bool:
  return any(c in s for c in ".*+?[]()|^$\\")


def _short_mj_name(name: str | None) -> str | None:
  if not name:
    return None
  return name.split("/", 1)[-1]


def _fill_gravity_frame_from_obj(
  model: Any, obj_type: int, obj_id: int, info: dict[str, Any]
) -> int | None:
  """Fill site/body/frame from a MuJoCo sensor obj/ref; return body id if known."""
  import mujoco

  if obj_id < 0:
    return None
  if obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
    site_name = _short_mj_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, obj_id))
    body_id = int(model.site_bodyid[obj_id])
    body_name = _short_mj_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
    if site_name:
      info["site_name"] = site_name
    if body_name:
      info["body_name"] = body_name
      info["frame"] = body_name.removesuffix("_link")
    return body_id
  if obj_type == int(mujoco.mjtObj.mjOBJ_BODY):
    body_name = _short_mj_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, obj_id))
    if body_name:
      info["body_name"] = body_name
      info["frame"] = body_name.removesuffix("_link")
    return obj_id
  return None


def _joint_chain_to_body(model: Any, body_id: int) -> list[str]:
  """Hinge/slide joints from floating base to ``body_id`` (root→…→IMU frame)."""
  import mujoco

  joints: list[str] = []
  bid = int(body_id)
  while bid > 0:
    jnt_adr = int(model.body_jntadr[bid])
    jnt_num = int(model.body_jntnum[bid])
    for k in range(jnt_num):
      jid = jnt_adr + k
      if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
        continue
      jname = _short_mj_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
      if jname:
        joints.append(jname)
    bid = int(model.body_parentid[bid])
  joints.reverse()
  return joints


def _resolve_projected_gravity_frame(
  env: ManagerBasedRlEnv, sensor_name: str | None
) -> dict[str, Any]:
  """Resolve gravity-direction body/site from training upvector sensor (if present)."""
  info: dict[str, Any] = {}
  if not sensor_name:
    return info
  info["sensor_name"] = sensor_name
  model = getattr(getattr(env, "sim", None), "mj_model", None)
  if model is None:
    return info
  import mujoco

  candidates = [sensor_name]
  short = sensor_name.split("/")[-1]
  if short and short != sensor_name:
    candidates.append(short)
  sid = -1
  for cand in candidates:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, cand)
    if sid >= 0:
      break
  if sid < 0:
    return info
  # Builtin framezaxis: IMU frame is ``obj`` (site/body); ``ref`` is often world.
  body_id = _fill_gravity_frame_from_obj(
    model, int(model.sensor_objtype[sid]), int(model.sensor_objid[sid]), info
  )
  if body_id is None:
    body_id = _fill_gravity_frame_from_obj(
      model, int(model.sensor_reftype[sid]), int(model.sensor_refid[sid]), info
    )
  if body_id is not None:
    info["joints"] = _joint_chain_to_body(model, body_id)
  return info


def _enrich_packed_amp_actor_frame(
  actor_blob: dict[str, Any],
  env_cfg: ManagerBasedRlEnvCfg,
) -> None:
  """Expand AMP packed ``frame`` into per-field obs scales for deployment."""
  terms = actor_blob.get("terms") or {}
  layout = actor_blob.get("layout") or []
  if list(terms.keys()) != ["frame"] and not (
    len(layout) == 1 and layout[0].get("name") == "frame"
  ):
    return

  jp = env_cfg.actions.get("joint_pos")
  if not isinstance(jp, JointPositionActionCfg):
    return
  # Infer DoF count from action scale vector / joint names when available.
  num_joints: int | None = None
  sc = jp.scale
  if isinstance(sc, (list, tuple)):
    num_joints = len(sc)
  else:
    joint_names = getattr(jp, "joint_names", None)
    if isinstance(joint_names, (list, tuple)) and joint_names:
      # Regex patterns still mean "all matching"; fall back to concat dim.
      num_joints = None

  hist = int(actor_blob.get("history_length") or 1)
  concat = actor_blob.get("concatenated_dim")
  if num_joints is None and isinstance(concat, int) and hist > 0:
    frame_dim = concat // hist
    # frame = 9 + 3 * J
    if frame_dim >= 9 and (frame_dim - 9) % 3 == 0:
      num_joints = (frame_dim - 9) // 3
  if num_joints is None or num_joints <= 0:
    return

  try:
    from src.tasks.amp.mdp.observations import (
      actor_frame_field_layout,
      actor_frame_obs_scales,
    )
  except ImportError:
    return

  fields = actor_frame_field_layout(num_joints)
  scales = actor_frame_obs_scales(num_joints)
  frame_dim = 9 + 3 * num_joints
  actor_blob["frame_dim"] = frame_dim
  actor_blob["obs_scales"] = scales
  # Keep ObservationManager term name ``frame`` for play.py filtering, but
  # expose logical fields (with baked-in scales) under terms and layout.
  frame_term = terms.setdefault("frame", {"scale": 1.0})
  if isinstance(frame_term, dict):
    frame_term["fields"] = {
      f["name"]: {"scale": f["scale"], "dim": f["dim"], "slice": f["slice"]}
      for f in fields
    }
  for row in layout:
    if row.get("name") == "frame":
      row["fields"] = fields
      row["frame_dim"] = frame_dim
      break
  else:
    actor_blob.setdefault("layout", []).append(
      {
        "name": "frame",
        "frame_dim": frame_dim,
        "fields": fields,
      }
    )


def build_deployment_alignment_snapshot(
  env_cfg: ManagerBasedRlEnvCfg,
  env: ManagerBasedRlEnv,
  *,
  task_id: str | None = None,
) -> dict[str, Any]:
  """Structured dict for YAML: joint order (via ``joints``), PD, default pose, obs/action scales."""
  out: dict[str, Any] = {
    "meta": {
      "kind": "deployment_alignment",
      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
      "framework": "mjlab",
    },
  }
  if task_id is not None:
    out["meta"]["task_id"] = task_id

  robot = env.scene["robot"]
  pd_map = collect_joint_pd_map(env_cfg, env)
  effort_map = collect_joint_effort_limit_map(env_cfg, env)
  frictionloss_map = collect_joint_frictionloss_map(env_cfg, env)
  joint_names: list[str] = list(robot.joint_names)
  active_names = set(joint_names)

  dj = getattr(robot.data, "default_joint_pos", None) if robot is not None else None
  hard_lim = getattr(robot.data, "joint_pos_limits", None) if robot is not None else None
  soft_lim = getattr(robot.data, "soft_joint_pos_limits", None) if robot is not None else None
  joints_rows: list[dict[str, Any]] = []
  for i, name in enumerate(joint_names):
    row: dict[str, Any] = {"name": name, "controlled": True}
    if dj is not None:
      row["default_joint_pos"] = float(dj[0, i])
    lim_src = hard_lim if hard_lim is not None else soft_lim
    if lim_src is not None:
      row["joint_pos_limits"] = [float(lim_src[0, i, 0]), float(lim_src[0, i, 1])]
    kp, kd = pd_map.get(name, (None, None))
    row["kp"] = kp
    row["kd"] = kd
    # Max torque clamp used in sim/deploy: τ ∈ [-effort_limit, +effort_limit].
    row["effort_limit"] = effort_map.get(name)
    row["frictionloss"] = frictionloss_map.get(name)
    joints_rows.append(row)

  entities = env_cfg.scene.entities
  robot_cfg = entities.get("robot") if entities else None
  init = getattr(robot_cfg, "init_state", None) if robot_cfg is not None else None

  # Unused / baked joints: exact names in init_state.joint_pos that are not actuated.
  # (Entity resolve_expr ignores them; deploy needs their default angles on full robot.)
  if init is not None and isinstance(getattr(init, "joint_pos", None), dict):
    for name, value in init.joint_pos.items():
      nm = str(name)
      if nm in active_names or _is_regex_pattern(nm):
        continue
      if not isinstance(value, Number):
        continue
      joints_rows.append(
        {
          "name": nm,
          "default_joint_pos": float(value),
          "kp": None,
          "kd": None,
          "controlled": False,
        }
      )

  out["joints"] = joints_rows

  ri: dict[str, Any] = {}
  if init is not None:
    ri["pos"] = _to_float_list(getattr(init, "pos", None))
    ri["rot"] = _to_float_list(getattr(init, "rot", None))
    # joint init pose is already per-joint in ``joints[].default_joint_pos`` (no duplicate map here).
  out["robot_init"] = ri if ri else None

  obs_out: dict[str, Any] = {}
  for group_name, group_cfg in env_cfg.observations.items():
    if group_name == "critic":
      continue
    if not isinstance(group_cfg, ObservationGroupCfg):
      continue
    terms_out: dict[str, Any] = {}
    for term_name, term_cfg in group_cfg.terms.items():
      scale = getattr(term_cfg, "scale", None)
      term_d: dict[str, Any] = {"scale": 1.0 if scale is None else scale}
      noise = getattr(term_cfg, "noise", None)
      if noise is not None:
        term_d["noise"] = _noise_to_yaml(noise)
      terms_out[term_name] = term_d
    obs_out[group_name] = {"history_length": group_cfg.history_length, "terms": terms_out}
  out["observations"] = obs_out

  # Actor observation layout (term order, flattened dims, scales, params) from live ObservationManager.
  gravity_frame: dict[str, Any] = {}
  om = getattr(env, "observation_manager", None)
  if om is not None:
    term_names = getattr(om, "_group_obs_term_names", {}).get("actor")
    term_dims = getattr(om, "_group_obs_term_dim", {}).get("actor")
    term_cfgs = getattr(om, "_group_obs_term_cfgs", {}).get("actor")
    if term_names is not None and term_dims is not None and term_cfgs is not None:
      layout: list[dict[str, Any]] = []
      for name, dims, tcfg in zip(term_names, term_dims, term_cfgs, strict=True):
        flat = int(np.prod(dims))
        params = _params_for_yaml(getattr(tcfg, "params", None))
        if name == "projected_gravity":
          gravity_frame = _resolve_projected_gravity_frame(
            env, params.get("sensor_name") if isinstance(params.get("sensor_name"), str) else None
          )
          # Keep layout params to frame identity only; joint chain lives under meta.imu_alignment.
          for key in ("body_name", "site_name", "frame"):
            if gravity_frame.get(key) and key not in params:
              params[key] = gravity_frame[key]
          if gravity_frame.get("sensor_name") and "sensor_name" not in params:
            params["sensor_name"] = gravity_frame["sensor_name"]
        layout.append(
          {
            "name": name,
            "shape": list(int(x) for x in dims),
            "flatten_dim": flat,
            "history_length": int(getattr(tcfg, "history_length", 0)),
            "scale": _scale_to_yaml(getattr(tcfg, "scale", None)),
            "noise": _noise_to_yaml(getattr(tcfg, "noise", None)),
            "params": params,
          }
        )
      actor_blob = out["observations"].setdefault("actor", {})
      actor_blob["layout"] = layout
      god = om.group_obs_dim.get("actor")
      if isinstance(god, tuple) and len(god) == 1:
        actor_blob["concatenated_dim"] = int(god[0])
      elif god is not None:
        actor_blob["concatenated_dim"] = int(np.prod(god))

      hs_yaml = _height_scan_sensor_yaml(env_cfg, env)
      if hs_yaml is not None:
        actor_blob["height_scan_sensor"] = hs_yaml

      # AMP packs base_ang_vel / gravity / cmd / q / dq / actions into one
      # ``frame`` term with baked-in scales (not ObservationTermCfg.scale).
      _enrich_packed_amp_actor_frame(actor_blob, env_cfg)

  if gravity_frame:
    imu_align: dict[str, Any] = {}
    # Ang vel sensor from layout if present.
    lay = out.get("observations", {}).get("actor", {}).get("layout") or []
    for row in lay:
      if row.get("name") == "base_ang_vel":
        sn = (row.get("params") or {}).get("sensor_name")
        if sn:
          imu_align["ang_vel"] = {
            "training_frame": "pelvis",
            "sensor_name": sn,
          }
        break
    joints = list(gravity_frame.get("joints") or [])
    if joints:
      desc = f"projected_gravity 关节: {', '.join(joints)}"
    else:
      frame = gravity_frame.get("frame") or gravity_frame.get("body_name") or "pelvis"
      desc = f"projected_gravity 无中间关节 (IMU 在 {frame})"
    pg: dict[str, Any] = {
      "training_frame": gravity_frame.get("frame") or gravity_frame.get("body_name"),
      "body_name": gravity_frame.get("body_name"),
      "site_name": gravity_frame.get("site_name"),
      "sensor_name": gravity_frame.get("sensor_name"),
      "joints": joints,
      "deploy_mode": "torso_fk",
      "description": desc,
    }
    imu_align["projected_gravity"] = {k: v for k, v in pg.items() if v is not None}
    out["meta"]["imu_alignment"] = imu_align

  jp = env_cfg.actions.get("joint_pos")
  if isinstance(jp, JointPositionActionCfg):
    sc = jp.scale
    if isinstance(sc, dict):
      out["actions"] = {"joint_pos": {"scale": {str(k): float(v) if isinstance(v, Number) else v for k, v in sc.items()}}}
    else:
      out["actions"] = {"joint_pos": {"scale": sc}}
  else:
    out["actions"] = {"joint_pos": {"note": f"not JointPositionActionCfg: {type(jp).__name__}"}}

  return out


def print_env_input_output_scales(
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: Any | None = None,
) -> None:
  """Log observation-term scales (policy input path) and joint action scales (output path).

  Also logs RL runner clip_actions and obs_normalization when ``agent_cfg`` is provided.
  """
  print("[INFO] --- Env observation scales (per term, after term func; default scale=1.0) ---")
  for group_name, group_cfg in env_cfg.observations.items():
    if not isinstance(group_cfg, ObservationGroupCfg):
      continue
    print(f"  [{group_name}] history_length={group_cfg.history_length}")
    for term_name, term_cfg in group_cfg.terms.items():
      scale = getattr(term_cfg, "scale", None)
      scale_s = "1.0" if scale is None else repr(scale)
      noise = getattr(term_cfg, "noise", None)
      noise_s = ""
      if noise is not None:
        n_min = getattr(noise, "n_min", None)
        n_max = getattr(noise, "n_max", None)
        if n_min is not None and n_max is not None:
          noise_s = f", noise_uniform=({n_min}, {n_max})"
        else:
          noise_s = f", noise={noise!r}"
      print(f"    - {term_name}: scale={scale_s}{noise_s}")

  print("[INFO] --- Env action scale (joint_pos policy output -> PD target offset) ---")
  jp = env_cfg.actions.get("joint_pos")
  if isinstance(jp, JointPositionActionCfg):
    sc = jp.scale
    if isinstance(sc, dict):
      for k in sorted(sc.keys()):
        print(f"  - {k}: {sc[k]}")
    else:
      print(f"  (scalar or uniform) scale={sc!r}")
  else:
    print(f"  [skip] joint_pos is not JointPositionActionCfg: {type(jp).__name__}")

  if agent_cfg is None:
    return

  clip = getattr(agent_cfg, "clip_actions", None)
  print(f"[INFO] --- RL wrapper --- clip_actions={clip!r}")
  actor = getattr(agent_cfg, "actor", None)
  critic = getattr(agent_cfg, "critic", None)
  if actor is not None:
    print(
      f"[INFO] --- RL actor obs_normalization={getattr(actor, 'obs_normalization', None)!r} ---"
    )
  if critic is not None:
    print(
      f"[INFO] --- RL critic obs_normalization={getattr(critic, 'obs_normalization', None)!r} ---"
    )


def print_robot_init_and_defaults(
  env_cfg: ManagerBasedRlEnvCfg,
  env: ManagerBasedRlEnv,
) -> None:
  """Log root pose + per-joint neutral pose for ``robot`` (same source as keyframe when cfg supplies joint_pos).

  Prints once: ``EntityCfg.init_state`` root ``pos``/``rot``, then ``default_joint_pos[0]``
  in ``joint_names`` order (obs / pose reward / PD offset reference).
  """
  entities = env_cfg.scene.entities
  if not entities or "robot" not in entities:
    print("[INFO] --- Robot init / defaults: no scene.entities['robot'] ---")
    return

  robot_cfg = entities["robot"]
  init = getattr(robot_cfg, "init_state", None)
  print("[INFO] --- Robot init (root pose + default_joint_pos[0]; neutral / keyframe ref) ---")
  if init is None:
    print("  init_state: None")
  else:
    print(f"  pos={getattr(init, 'pos', None)!r}")
    print(f"  rot={getattr(init, 'rot', None)!r}")
    jp = getattr(init, "joint_pos", None)
    if jp is None:
      print(
        "  joint_pos: None → joints from XML/model keyframe; values below are resolved runtime default_joint_pos"
      )

  robot = env.scene["robot"]
  dj = getattr(robot.data, "default_joint_pos", None)
  names = getattr(robot, "joint_names", None)
  if dj is None or names is None:
    print("[INFO] --- default_joint_pos unavailable ---")
    return

  row = dj[0]
  for i, name in enumerate(names):
    print(f"  {name}: {float(row[i]):.6f}")
