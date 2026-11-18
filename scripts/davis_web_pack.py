"""DAVIS depth-shooting helpers for web_demo packing (ONNX + scene)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Match src.tasks.davis.mdp.observations proprio layout (22-DoF + neck).
DAVIS_FRAME_DIM = 72
DAVIS_HISTORY = 5
DAVIS_PROPRIO_DIM = DAVIS_FRAME_DIM * DAVIS_HISTORY  # 360
DAVIS_DEPTH_HW = (80, 168)
DAVIS_DEPTH_RENDER_HW = (94, 168)
DAVIS_FOVY_DEG = 60.0
DAVIS_HEAD_BODY = "head_camera_link"
# Match src.tasks.davis.robot_wbot_neck + CameraSensorCfg(zed2i_depth).
# Camera is NOT at body origin — offset on head_camera_link, moves with neck.
DAVIS_CAM_POS = (0.06, 0.0, 0.04)
# MuJoCo cam: look −Z, up +Y. Quat maps to body +X look / +Z up.
DAVIS_CAM_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)


def enrich_davis_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
  """Synthesize ``observations.actor`` for DAVIS runs (proprio groups only in YAML)."""
  joints = [j for j in alignment.get("joints", []) if j.get("controlled", True)]
  n = len(joints)
  if n != 22:
    raise ValueError(f"DAVIS web pack expects 22 controlled joints, got {n}")

  # Match play/Viser home (robot_wbot_neck): neck_yaw=0, neck_pitch=0.35.
  for j in joints:
    if j.get("name") == "neck_yaw_joint":
      j["default_joint_pos"] = 0.0
    elif j.get("name") == "neck_pitch_joint":
      j["default_joint_pos"] = 0.35

  # Per-frame fields; history stacked by browser (same as AMP stacked layout).
  fields = [
    {"name": "base_ang_vel", "dim": 3, "slice": [0, 3], "scale": 0.2},
    {"name": "projected_gravity", "dim": 3, "slice": [3, 6], "scale": 1.0},
    {"name": "joint_pos", "dim": n, "slice": [6, 6 + n], "scale": 1.0},
    {"name": "joint_vel", "dim": n, "slice": [6 + n, 6 + 2 * n], "scale": 0.05},
    {"name": "actions", "dim": n, "slice": [6 + 2 * n, 6 + 3 * n], "scale": 1.0},
  ]
  obs = alignment.setdefault("observations", {})
  # Keep actor_proprio / actor_depth metadata; add deployable actor blob.
  obs["actor"] = {
    "history_length": DAVIS_HISTORY,
    "frame_dim": DAVIS_FRAME_DIM,
    "concatenated_dim": DAVIS_PROPRIO_DIM,
    "fields": fields,
    "obs_scales": {
      "base_ang_vel": 0.2,
      "projected_gravity": 1.0,
      "joint_pos": 1.0,
      "joint_vel": 0.05,
      "actions": 1.0,
    },
    "depth": {
      "height": DAVIS_DEPTH_HW[0],
      "width": DAVIS_DEPTH_HW[1],
      "render_height": DAVIS_DEPTH_RENDER_HW[0],
      "render_width": DAVIS_DEPTH_RENDER_HW[1],
      "fovy_deg": DAVIS_FOVY_DEG,
      "head_body": DAVIS_HEAD_BODY,
      "cam_pos": list(DAVIS_CAM_POS),
      "cam_quat_wxyz": list(DAVIS_CAM_QUAT_WXYZ),
      "depth_min_m": 0.3,
      "depth_max_m": 5.0,
    },
  }
  meta = alignment.setdefault("meta", {})
  meta["play_profile"] = "davis"
  meta["obs_inputs"] = ["proprio", "depth"]
  return alignment


def build_davis_actor_for_export(agent_cfg: dict[str, Any], act_dim: int = 22):
  """Construct ``DavisActorModel`` matching train obs groups for ONNX export."""
  import torch
  from tensordict import TensorDict

  from src.tasks.davis.rl.davis_actor import DavisActorModel

  h, w = DAVIS_DEPTH_HW
  obs = TensorDict(
    {
      "actor_proprio": torch.zeros(1, DAVIS_PROPRIO_DIM),
      "actor_depth": torch.zeros(1, 1, h, w),
      "aux_gt": torch.zeros(1, 8),
    },
    batch_size=[1],
  )
  actor_cfg = agent_cfg.get("actor") or {}
  return DavisActorModel(
    obs,
    {"actor": ["actor_proprio", "actor_depth", "aux_gt"]},
    "actor",
    act_dim,
    hidden_dims=tuple(actor_cfg.get("hidden_dims") or (512, 256, 128)),
    activation=str(actor_cfg.get("activation") or "elu"),
    obs_normalization=bool(actor_cfg.get("obs_normalization", True)),
    distribution_cfg=actor_cfg.get("distribution_cfg"),
    anneal_k0=int(actor_cfg.get("anneal_k0", 1000)),
    anneal_k1=int(actor_cfg.get("anneal_k1", 8000)),
    anneal_ema_threshold=float(actor_cfg.get("anneal_ema_threshold", 0.5)),
    aux_geo_weight_ball=float(actor_cfg.get("aux_geo_weight_ball", 1.0)),
    aux_geo_weight_goal=float(actor_cfg.get("aux_geo_weight_goal", 1.0)),
    aux_vis_weight=float(actor_cfg.get("aux_vis_weight", 0.2)),
    deployment_beta=1.0,
  )


def write_davis_robot_xml(pack_dir: Path) -> Path:
  """Dump 20-DoF wbot + neck MJCF into the pack (meshes via shared folder)."""
  from src.tasks.davis.robot_wbot_neck import get_spec_with_neck

  spec = get_spec_with_neck()
  xml = spec.to_xml()
  # Pack-local meshdir.
  xml = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes/"', xml, count=1)
  out = pack_dir / "robot_davis.xml"
  out.write_text(xml)
  return out


def build_davis_scene_xml(alignment: dict[str, Any], robot_xml: Path) -> str:
  """Floor + ball + goal frame + davis robot with position actuators."""
  robot_body = robot_xml.read_text()
  inner = re.sub(r"^\s*<mujoco[^>]*>", "", robot_body, count=1, flags=re.I)
  inner = re.sub(r"</mujoco>\s*$", "", inner, count=1, flags=re.I)
  inner = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes/"', inner, count=1)

  joints = [j for j in alignment["joints"] if j.get("controlled", True)]
  # Match penalty-kick default: goal@0, ball@+1.5x, robot behind ball facing goal (−x).
  # deployment_alignment robot_init is the standing default at origin — wrong for shoot.
  ball_x = 1.5
  robot_x = 3.2
  robot_z = 0.59
  # yaw=π → face −x (toward goal); MuJoCo quat wxyz.
  inner = re.sub(
    r'(<body name="pelvis"\s+)([^>]*)(>)',
    rf'\1pos="{robot_x} 0 {robot_z}" quat="0 0 0 1"\3',
    inner,
    count=1,
    flags=re.I,
  )

  # Inject training zed2i_depth camera on head_camera_link (moves with neck).
  # Same pos/quat/fovy as CameraSensorCfg in env_cfg.
  cam_pos = " ".join(str(x) for x in DAVIS_CAM_POS)
  cam_quat = " ".join(str(x) for x in DAVIS_CAM_QUAT_WXYZ)
  cam_tag = (
    f'\n            <camera name="zed2i_depth" pos="{cam_pos}" quat="{cam_quat}" '
    f'fovy="{DAVIS_FOVY_DEG}" resolution="{DAVIS_DEPTH_RENDER_HW[1]} '
    f'{DAVIS_DEPTH_RENDER_HW[0]}"/>'
  )
  if 'name="zed2i_depth"' not in inner:
    # Insert after head_camera_link opening (or after its first child site/geom).
    replaced = False
    def _inject_cam(m: re.Match[str]) -> str:
      nonlocal replaced
      replaced = True
      return m.group(0) + cam_tag

    inner2 = re.sub(
      r'(<body name="head_camera_link"[^>]*>)',
      _inject_cam,
      inner,
      count=1,
      flags=re.I,
    )
    if replaced:
      inner = inner2
    else:
      print("[WARN] head_camera_link not found — zed2i_depth camera not injected")

  floor_assets = """
  <texture type="2d" name="groundplane" builtin="checker" mark="edge"
           rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
           width="512" height="512"/>
  <material name="groundplane" texture="groundplane" texuniform="true"
            texrepeat="5 5" reflectance="0.1"/>
  <material name="ball_mat" rgba="1.0 0.45 0.1 1"/>
  <material name="goal_mat" rgba="0.92 0.92 0.95 1"/>
"""
  if re.search(r"<asset>", inner, flags=re.I):
    inner = re.sub(r"(<asset>)", r"\1" + floor_assets, inner, count=1, flags=re.I)
  else:
    inner = f"  <asset>{floor_assets}  </asset>\n" + inner

  # Goal frame matches get_goal_spec: posts along Z, crossbar quat rotates axis → +Y.
  extras = f"""
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"
          friction="1 0.005 0.0001"/>
    <body name="ball" pos="{ball_x} 0 0.11">
      <freejoint name="ball_freejoint"/>
      <geom name="ball_geom" type="sphere" size="0.11" material="ball_mat"
            mass="0.43" friction="0.8 0.01 0.01"/>
    </body>
    <body name="goal" pos="0 0 0">
      <geom name="goal_post_l" type="cylinder" size="0.05 0.9"
            pos="0 1.2 0.9" material="goal_mat" contype="0" conaffinity="0"/>
      <geom name="goal_post_r" type="cylinder" size="0.05 0.9"
            pos="0 -1.2 0.9" material="goal_mat" contype="0" conaffinity="0"/>
      <geom name="goal_crossbar" type="cylinder" size="0.05 1.2"
            pos="0 0 1.8" quat="0.70710678 0.70710678 0 0" material="goal_mat"
            contype="0" conaffinity="0"/>
    </body>
"""
  if not re.search(r"</worldbody>", inner, flags=re.I):
    raise ValueError("DAVIS robot MJCF missing </worldbody>")
  inner = re.sub(r"</worldbody>", extras + "  </worldbody>", inner, count=1, flags=re.I)

  # Match training BuiltinPositionActuatorCfg: armature + frictionloss required
  # for implicitfast PD stability (without these the browser robot falls instantly).
  try:
    from pack_web_demo import _inject_joint_armature, _inject_joint_frictionloss
  except ImportError:
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from pack_web_demo import _inject_joint_armature, _inject_joint_frictionloss

  joint_names = [str(j["name"]) for j in joints]
  inner = _inject_joint_armature(inner, joint_names, family="wbot")
  inner = _inject_joint_frictionloss(inner, joints)

  actuator_lines = ["  <actuator>"]
  for j in joints:
    name = j["name"]
    kp = float(j.get("kp") or 20.0)
    kd = float(j.get("kd") or 1.0)
    limits = j.get("joint_pos_limits") or [-3.14, 3.14]
    lo, hi = float(limits[0]), float(limits[1])
    effort = float(j.get("effort_limit") or 50.0)
    actuator_lines.append(
      f'    <position name="{name}" joint="{name}" kp="{kp}" kv="{kd}" '
      f'ctrlrange="{lo} {hi}" forcerange="{-effort} {effort}"/>'
    )
  actuator_lines.append("  </actuator>")
  actuators = "\n".join(actuator_lines)

  return f"""<mujoco model="davis_web_demo">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.3 0.3 0.3"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
{inner}
{actuators}
</mujoco>
"""
