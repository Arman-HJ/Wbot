"""Walk AMP + Kick + Roll + Get Up policies in one Viser session (ball in scene).

- Start in **walk** (stand still until you move vx/vy/wz).
- **Kick**: switch policy, preserving robot and ball state.
- **Walk**: switch back to walk policy.
- **Roll**: roll from supine to prone, then hand off to Get Up without resetting.
- **Get Up**: run the independent prone reference, then return to Walk if upright.
- **Prone → Get Up**: explicitly place the robot in the trained test pose.
- Kick reference advances once, then holds the last frame until Walk is pressed.
"""

from __future__ import annotations

import os
from copy import deepcopy
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import tyro
import viser

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import (
  quat_apply_inverse, quat_apply, quat_mul, quat_inv, yaw_quat, matrix_from_quat,
  quat_from_euler_xyz,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.viewer import ViserPlayViewer
from src.tasks.kick.mdp.events import _init_kick_ball_tracking
from src.viewer.gamepad import BrowserGamepad
from src.viewer.force_pulse import ForcePulse
from src.viewer.fall_guard import FallGuard
from src.viewer.control_latency import ControlLatency

_WBOT_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)

_STAND_CMD_EPS = 0.05


def _load_walk_deployment_alignment(walk_checkpoint: Path) -> dict:
  """Read ``params/deployment_alignment.yaml`` next to the walk checkpoint."""
  align_path = walk_checkpoint.parent / "params" / "deployment_alignment.yaml"
  if not align_path.is_file():
    return {}
  try:
    import yaml

    data = yaml.safe_load(align_path.read_text())
  except Exception as exc:  # noqa: BLE001
    print(f"[WARN] Could not parse {align_path}: {exc}")
    return {}
  return data if isinstance(data, dict) else {}


def _action_scale_dict_to_buf(
  scale: dict[str, float] | float,
  joint_names: tuple[str, ...],
  template: torch.Tensor,
) -> torch.Tensor:
  """Expand a joint-name / regex scale map onto the env action scale buffer."""
  out = torch.ones_like(template)
  if isinstance(scale, dict):
    ids, _, values = resolve_matching_names_values(scale, list(joint_names))
    if len(ids) != len(joint_names):
      raise ValueError(
        f"Walk action scale must cover all {len(joint_names)} joints, got {len(ids)}"
      )
    out[:, ids] = torch.tensor(values, device=template.device, dtype=template.dtype)
  else:
    out.fill_(float(scale))
  return out


def _joint_pos_map_to_buf(
  joint_pos: dict[str, float],
  joint_names: tuple[str, ...],
  template: torch.Tensor,
) -> torch.Tensor:
  """Expand init_state / home joint_pos (names or regex) onto an action buffer."""
  ids, _, values = resolve_matching_names_values(joint_pos, list(joint_names))
  if len(ids) != len(joint_names):
    raise ValueError(
      f"Home joint_pos must cover all {len(joint_names)} joints, got {len(ids)}"
    )
  out = torch.zeros_like(template)
  out[:, ids] = torch.tensor(values, device=template.device, dtype=template.dtype)
  return out


def _walk_home_from_alignment(align: dict) -> dict[str, float] | None:
  joints = align.get("joints")
  if not isinstance(joints, list):
    return None
  out: dict[str, float] = {}
  for row in joints:
    if not isinstance(row, dict):
      continue
    name = row.get("name")
    val = row.get("default_joint_pos")
    if isinstance(name, str) and val is not None:
      out[name] = float(val)
  return out or None


@dataclass
class WalkKickPlayConfig:
  walk_checkpoint: str
  kick_checkpoint: str
  getup_checkpoint: str
  roll_checkpoint: str
  walk_task: str = "AMP-Walk-Run-Wbot-20DOF"
  kick_task: str = "Wbot-20Dof-Kick"
  getup_task: str = "Wbot-20Dof-Tracking-Getup"
  roll_task: str = "Wbot-20Dof-Tracking-RollToProne"
  num_envs: int = 1
  device: str | None = None
  viser_host: str = "0.0.0.0"
  viser_port: int = 8095
  no_terminations: bool = True
  smoke_test: bool = False
  """Run finite walk / kick / reset checks without opening a viewer."""
  push_smoke_test: bool = False
  """Validate a finite torso push without running the full policy regression."""
  ball_rolling_friction: float = 0.015
  """Rolling friction length (m); larger values stop the ball sooner."""
  stand_hold: bool = False
  """Optional PD home hold at zero command; default runs the trained policy."""


class WalkActorMlp(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(345, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 20),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


def _load_walk_actor(path: Path, device: str) -> WalkActorMlp:
  ckpt = torch.load(path, map_location=device, weights_only=False)
  actor = WalkActorMlp().to(device)
  actor.load_state_dict(
    {k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")},
    strict=True,
  )
  actor.eval()
  print(f"[INFO] Loaded walk actor from {path.name}")
  return actor


def _joint_pos_action_scale_buf(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  buf = getattr(term, "_scale", None)
  if buf is None:
    buf = term.scale
  assert torch.is_tensor(buf)
  return buf


def _ordered_joint_ids(robot) -> list[int]:
  name_to_id = {n: i for i, n in enumerate(robot.joint_names)}
  return [name_to_id[n] for n in _WBOT_JOINT_NAMES]


def snap_robot_to_walk_play_pose(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Reset the robot to the shared training home pose."""
  robot = env.scene["robot"]
  device = env.device
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
  default_q = robot.data.default_joint_pos
  assert default_q is not None
  robot.write_joint_state_to_sim(
    default_q[env_ids].clone(),
    torch.zeros_like(default_q[env_ids]),
    env_ids=env_ids,
  )
  n = len(env_ids)
  root = torch.zeros(n, 13, device=device)
  if robot.data.default_root_state is not None:
    root[:, :7] = robot.data.default_root_state[env_ids, :7]
  else:
    root[:, 2] = 0.59
    root[:, 3] = 1.0
  root[:, :3] += env.scene.env_origins[env_ids]
  robot.write_root_state_to_sim(root, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


snap_robot_to_default_stand = snap_robot_to_walk_play_pose


def aligned_motion_orientation(env, command_name: str) -> torch.Tensor:
  motion = env.command_manager.get_term(command_name)
  target = motion.anchor_quat_w
  mode = {"motion": "kick", "getup_motion": "getup", "roll_motion": "roll"}[command_name]
  heading_name = f"_{mode}_heading_delta"
  if hasattr(env, heading_name):
    target = quat_mul(getattr(env, heading_name), target)
  relative = quat_mul(quat_inv(motion.robot_anchor_quat_w), target)
  return matrix_from_quat(relative)[..., :2].reshape(env.num_envs, 6)


def place_ball_only(env, env_ids=None) -> None:
  """Transform the training ball position by the robot's current XY and yaw."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot = env.scene["robot"]
  motion = env.command_manager.get_term("motion")
  root_idx = list(motion.cfg.body_names).index("pelvis")
  ref_pos = motion.motion.body_pos_w[0, root_idx]
  ref_quat = motion.motion.body_quat_w[0, root_idx].expand(env.num_envs, -1)
  delta = quat_mul(yaw_quat(robot.data.root_link_quat_w), quat_inv(yaw_quat(ref_quat)))
  env._kick_heading_delta = delta
  offset = torch.tensor(motion.cfg.ball_pos, device=env.device) - ref_pos
  pos = robot.data.root_link_pos_w + quat_apply(delta, offset.expand(env.num_envs, -1))
  pos[:, 2] = motion.cfg.ball_pos[2] + env.scene.env_origins[:, 2]
  state = torch.zeros(len(env_ids), 13, device=env.device)
  state[:, :3] = pos[env_ids]
  state[:, 3] = 1.0
  env.scene["ball"].write_root_state_to_sim(state, env_ids=env_ids)
  _init_kick_ball_tracking(env, env_ids, state[:, :3], yaw_quat(robot.data.root_link_quat_w[env_ids]))
  env.scene.write_data_to_sim()
  env.sim.forward()


def disable_motion_rsi(
  env: ManagerBasedRlEnv, command_name: str = "motion", active_mode: str = "kick"
) -> None:
  """Advance only the selected reference; never write clip poses onto the robot."""
  motion = env.command_manager.get_term(command_name)
  motion.cfg.debug_vis = False

  def _resample_no_rsi(env_ids: torch.Tensor) -> None:
    motion.time_steps[env_ids] = int(getattr(motion.cfg, "start_time_step", 0) or 0)
    motion.update_relative_body_poses()

  def _update_time_only() -> None:
    if getattr(env, "_dual_mode", "walk") == active_mode:
      motion.time_steps.add_(1).clamp_(max=motion.motion.time_step_total - 1)
    motion.update_relative_body_poses()

  motion._resample_command = _resample_no_rsi  # type: ignore[method-assign]
  motion._update_command = _update_time_only  # type: ignore[method-assign]
  motion.time_steps.zero_()
  print(f"[INFO] {active_mode}: independent reference clock, no automatic robot reset")


class MultiPolicyController:
  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    kick_policy,
    walk_actor: WalkActorMlp,
    device: str,
    getup_policy,
    getup_scale,
    roll_policy,
    roll_scale,
    walk_scale,
    walk_home: dict[str, float],
    stand_hold: bool = False,
  ) -> None:
    self.env = env
    self.raw = env.unwrapped
    self.kick_policy = kick_policy
    self.walk_actor = walk_actor
    self.getup_policy = getup_policy
    self.roll_policy = roll_policy
    self.device = device
    self.stand_hold = stand_hold
    self.mode: Literal["walk", "kick", "getup", "roll"] = "walk"
    self.twist = torch.zeros(1, 3, device=device)
    self.history = torch.zeros(self.raw.num_envs, 5, 69, device=device)
    # Kick/getup/roll share legacy home + armature scales. Walk AMP uses stand
    # home and its own deployment_alignment scales (same armature recipe now).
    self._scale_buf = _joint_pos_action_scale_buf(self.raw)
    self._kick_scales = self._scale_buf.detach().clone()
    term = self.raw.action_manager.get_term("joint_pos")
    self._kick_offset = term._offset.detach().clone()
    self._walk_scales = _action_scale_dict_to_buf(
      walk_scale, _WBOT_JOINT_NAMES, self._scale_buf
    )
    self._walk_offset = _joint_pos_map_to_buf(
      walk_home, _WBOT_JOINT_NAMES, self._kick_offset
    )
    for mode, scale in (("getup", getup_scale), ("roll", roll_scale)):
      scales = torch.ones_like(self._scale_buf)
      if isinstance(scale, dict):
        ids, _, values = resolve_matching_names_values(scale, _WBOT_JOINT_NAMES)
        if len(ids) != 20:
          raise ValueError(f"{mode} scale must specify all 20 joints")
        scales[:, ids] = torch.tensor(values, device=device)
      else:
        scales.fill_(float(scale))
      setattr(self, f"_{mode}_scales", scales)
    self.joint_ids = _ordered_joint_ids(self.raw.scene["robot"])
    self.torso_id = list(self.raw.scene["robot"].body_names).index("torso_link")
    self._status = "walk (stand)"
    print("[INFO] Action order:", term._target_names)
    if tuple(term._target_names) != _WBOT_JOINT_NAMES:
      raise ValueError("Joint action order differs from the trained WBOT policy")
    print(
      "[INFO] Walk action scale (min/max): "
      f"{float(self._walk_scales.min()):.3f}/{float(self._walk_scales.max()):.3f}; "
      "kick/getup keep legacy home, walk uses stand home for obs + PD offset"
    )
    self.fall_guard = FallGuard(self.raw)
    self.latency = ControlLatency(self)
    self.reset_walk_history()

  def reset_walk_history(self) -> None:
    # Replicate the current frame for a fresh, frame-major history.
    frame = self._build_walk_frame()
    self.history[:] = frame.unsqueeze(1).expand(-1, 5, -1)

  def place_ball(self) -> None:
    place_ball_only(self.raw)

  def _install_action_home(self, scales: torch.Tensor, offset: torch.Tensor, *, keep_q_des: bool) -> None:
    """Switch joint_pos scale + PD offset; optionally preserve absolute q_des."""
    term = self.raw.action_manager.get_term("joint_pos")
    if keep_q_des:
      q_des = term._offset + self._scale_buf * self.raw.action_manager.action
    self._scale_buf[:] = scales
    term._offset[:] = offset
    self.raw.scene["robot"].data.default_joint_pos[:, self.joint_ids] = offset
    if keep_q_des:
      self.raw.action_manager.action[:] = (q_des - offset) / scales.clamp_min(1e-6)

  def _set_walk_action_scale(self) -> None:
    self._install_action_home(self._walk_scales, self._walk_offset, keep_q_des=False)

  def _set_kick_action_scale(self) -> None:
    self._install_action_home(self._kick_scales, self._kick_offset, keep_q_des=False)

  def enter_kick(self) -> None:
    """Switch policy without moving the robot or ball."""
    self.fall_guard.restore()
    self._set_kick_action_scale()
    motion = self.raw.command_manager.get_term("motion")
    motion.time_steps.zero_()
    robot = self.raw.scene["robot"]
    root_idx = list(motion.cfg.body_names).index("pelvis")
    reference = motion.motion.body_quat_w[0, root_idx].expand(self.raw.num_envs, -1)
    self.raw._kick_heading_delta = quat_mul(
      yaw_quat(robot.data.root_link_quat_w), quat_inv(yaw_quat(reference))
    )
    self.mode = "kick"
    self.raw._dual_mode = "kick"
    self.raw.action_manager.reset()
    self._status = "kick"
    self.latency.reset()

  def enter_getup(self) -> None:
    self._enter_recovery("getup")

  def enter_roll(self) -> None:
    self._enter_recovery("roll")

  def _enter_recovery(self, mode: Literal["getup", "roll"]) -> None:
    """Start the selected reference without teleporting the robot or ball."""
    self.fall_guard.restore()
    # Getup/roll were trained against the same legacy home as kick.
    self._install_action_home(
      getattr(self, f"_{mode}_scales"), self._kick_offset, keep_q_des=False
    )
    motion = self.raw.command_manager.get_term(f"{mode}_motion")
    motion.time_steps.zero_()
    robot = self.raw.scene["robot"]
    root_idx = list(motion.cfg.body_names).index("pelvis")
    reference = motion.motion.body_quat_w[0, root_idx].expand(self.raw.num_envs, -1)
    # Use the relative rotation: absolute Euler yaw is undefined for the
    # supine reference (its pelvis X axis is vertical).
    setattr(self.raw, f"_{mode}_heading_delta", yaw_quat(
      quat_mul(robot.data.root_link_quat_w, quat_inv(reference))
    ))
    self.raw.action_manager.reset()
    self.twist.zero_()
    self.mode = mode
    self.raw._dual_mode = mode
    self._status = "GET UP" if mode == "getup" else "ROLL"
    self.latency.reset()

  def demo_getup(self, mode: Literal["getup", "roll"] = "getup") -> None:
    """Explicit test: put the robot in the selected reference pose, keep ball."""
    robot = self.raw.scene["robot"]
    motion = self.raw.command_manager.get_term(f"{mode}_motion")
    root_idx = list(motion.cfg.body_names).index("pelvis")
    n = self.raw.num_envs
    reference = motion.motion.body_quat_w[0, root_idx].expand(n, -1)
    delta = quat_mul(yaw_quat(robot.data.root_link_quat_w), quat_inv(yaw_quat(reference)))
    state = torch.zeros(n, 13, device=self.device)
    state[:, :2] = robot.data.root_link_pos_w[:, :2]
    state[:, 2] = motion.motion.body_pos_w[0, root_idx, 2] + self.raw.scene.env_origins[:, 2]
    state[:, 3:7] = quat_mul(delta, reference)
    state[:, 7:10] = quat_apply(delta, motion.motion.body_lin_vel_w[0, root_idx].expand(n, -1))
    state[:, 10:13] = quat_apply(delta, motion.motion.body_ang_vel_w[0, root_idx].expand(n, -1))
    robot.write_joint_state_to_sim(
      motion.motion.joint_pos[0].expand(n, -1).clone(),
      motion.motion.joint_vel[0].expand(n, -1).clone(),
    )
    robot.write_root_state_to_sim(state)
    robot.reset()
    self.raw.scene.write_data_to_sim()
    self.raw.sim.forward()
    self._enter_recovery(mode)

  def enter_walk(self) -> None:
    self.fall_guard.restore()
    keep = self.mode != "walk"
    self._install_action_home(self._walk_scales, self._walk_offset, keep_q_des=keep)
    self.mode = "walk"
    self.raw._dual_mode = "walk"
    self.twist.zero_()
    self.reset_walk_history()
    self._status = "walk (stand)"
    self.latency.reset()

  def reset(self) -> None:
    # Snap to AMP stand home (walk default), not kick/getup legacy home.
    self._install_action_home(self._walk_scales, self._walk_offset, keep_q_des=False)
    snap_robot_to_walk_play_pose(self.raw)
    place_ball_only(self.raw)
    self.raw.action_manager.reset()
    for name in ("_kick_heading_delta", "_getup_heading_delta", "_roll_heading_delta"):
      if hasattr(self.raw, name):
        delattr(self.raw, name)
    self.enter_walk()

  def _build_walk_frame(self) -> torch.Tensor:
    """Match DroidUp ``amp.mdp.observations.actor_frame`` (69-D)."""
    robot = self.raw.scene["robot"]
    quat = robot.data.body_link_quat_w[:, self.torso_id, :]
    ang_w = robot.data.body_link_ang_vel_w[:, self.torso_id, :]
    ang_b = quat_apply_inverse(quat, ang_w)
    gravity_b = quat_apply_inverse(quat, robot.data.gravity_vec_w)
    ids = self.joint_ids
    q = robot.data.joint_pos[:, ids]
    dq = robot.data.joint_vel[:, ids]
    # Always use AMP stand home (not kick legacy default_joint_pos).
    default = self._walk_offset
    return torch.cat(
      (
        0.2 * ang_b,
        gravity_b,
        self.twist.expand(q.shape[0], -1),
        q - default,
        0.05 * dq,
        self.raw.action_manager.action,
      ),
      dim=1,
    )

  def __call__(self, obs) -> torch.Tensor:
    if self.fall_guard.check(self.mode):
      self.twist.zero_()
      self.raw._dual_mode = "disabled"
      self._status = self.fall_guard.reason
      self.latency.reset()
      return torch.zeros(self.raw.num_envs, 20, device=self.device)
    return self.latency.policy(self._compute_action(obs))

  def _compute_action(self, obs) -> torch.Tensor:
    if self.mode == "roll":
      motion = self.raw.command_manager.get_term("roll_motion")
      robot = self.raw.scene["robot"]
      done = bool((motion.time_steps >= motion.motion.time_step_total - 1).all())
      gravity = quat_apply_inverse(motion.robot_anchor_quat_w, robot.data.gravity_vec_w)
      target = quat_apply_inverse(motion.anchor_quat_w, robot.data.gravity_vec_w)
      prone = bool(((gravity - target).norm(dim=-1) < 0.65).all())
      if done and prone:
        self.enter_getup()
        obs = self.env.get_observations()
      else:
        self._status = f"ROLL frame={int(motion.time_steps[0])}/{motion.motion.time_step_total}"
        if done:
          self._status += " (reference ended; not prone)"
        return self.roll_policy({"actor": obs["roll_actor"]})
    if self.mode == "getup":
      motion = self.raw.command_manager.get_term("getup_motion")
      robot = self.raw.scene["robot"]
      done = bool((motion.time_steps >= motion.motion.time_step_total - 1).all())
      gravity = quat_apply_inverse(robot.data.root_link_quat_w, robot.data.gravity_vec_w)
      upright = bool(((robot.data.root_link_pos_w[:, 2] - self.raw.scene.env_origins[:, 2] > 0.45) & (gravity[:, 2] < -0.8)).all())
      if done and upright:
        self.enter_walk()
      else:
        self._status = f"GET UP frame={int(motion.time_steps[0])}/{motion.motion.time_step_total}"
        if done:
          self._status += " (reference ended; not upright)"
        return self.getup_policy({"actor": obs["getup_actor"]})
    if self.mode == "kick":
      motion = self.raw.command_manager.get_term("motion")
      actions = self.kick_policy(obs)
      self._status = (
        f"kick  frame={int(motion.time_steps[0].item())}/"
        f"{int(motion.motion.time_step_total)}"
      )
      return actions

    # Run the actor at zero command too, unless PD hold was requested.
    frame = self._build_walk_frame()
    self.history = torch.cat((self.history[:, 1:], frame.unsqueeze(1)), dim=1)
    cmd_mag = float(self.twist.abs().sum())
    if self.stand_hold and cmd_mag < _STAND_CMD_EPS:
      self._status = "WALK stand (a=0 / PD home)"
      return torch.zeros(self.raw.num_envs, 20, device=self.device)

    with torch.inference_mode():
      walk_actions = self.walk_actor(self.history.reshape(self.raw.num_envs, -1))
    self._status = (
      f"WALK  cmd=({self.twist[0,0]:.2f},{self.twist[0,1]:.2f},{self.twist[0,2]:.2f})"
    )
    return walk_actions


class WalkKickViewer(ViserPlayViewer):
  controller: MultiPolicyController

  def setup(self) -> None:
    super().setup()
    # Policy passed to the viewer IS the dual controller.
    if not hasattr(self, "controller") or self.controller is None:
      self.controller = self.policy  # type: ignore[assignment]
    # Enter walk immediately — do not wait for a Walk button click.
    self.controller.enter_walk()
    if self._is_paused:
      self.request_toggle_pause()

    with self._server.gui.add_folder("Walk / Kick / Roll / Get Up"):
      self._mode_html = self._server.gui.add_html(
        "<b>WALK（已自动进入）</b><br/>"
        "<small>Roll→Get Up→Walk 自动接续；Kick 保留球位置</small>"
      )
      kick_btn = self._server.gui.add_button(
        "Kick", icon=viser.Icon.BALL_AMERICAN_FOOTBALL
      )
      walk_btn = self._server.gui.add_button("Walk", icon=viser.Icon.WALK)
      ball_btn = self._server.gui.add_button("Reset Ball")
      roll_btn = self._server.gui.add_button("Roll（翻身→起立）")
      roll_demo_btn = self._server.gui.add_button("Supine → Roll（仰卧翻身测试）")

      @roll_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"dual_mode": "roll"})

      @roll_demo_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"roll_demo": True})

      getup_btn = self._server.gui.add_button("Get Up（原地起立）")
      demo_btn = self._server.gui.add_button("Prone → Get Up（趴下并起立测试）")

      @getup_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"dual_mode": "getup"})

      @demo_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"getup_demo": True})

      @ball_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"reset_ball": True})

      self._vx = self._server.gui.add_slider("vx (m/s)", -0.8, 1.2, 0.02, 0.0)
      self._vy = self._server.gui.add_slider("vy (m/s)", -0.5, 0.5, 0.02, 0.0)
      self._wz = self._server.gui.add_slider("wz (rad/s)", -1.5, 1.5, 0.05, 0.0)

      @kick_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"dual_mode": "kick"})
        self._mode_html.content = "<b>mode:</b> kick（保留球的位置）"

      @walk_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"dual_mode": "walk"})
        self._vx.value = 0.0
        self._vy.value = 0.0
        self._wz.value = 0.0
        self._mode_html.content = "<b>WALK</b>"

      @self._vx.on_update
      def _(_) -> None:
        self.request_action("CUSTOM", {"twist": (0, float(self._vx.value))})

      @self._vy.on_update
      def _(_) -> None:
        self.request_action("CUSTOM", {"twist": (1, float(self._vy.value))})

      @self._wz.on_update
      def _(_) -> None:
        self.request_action("CUSTOM", {"twist": (2, float(self._wz.value))})

    with self._server.gui.add_folder("摔倒保护"):
      guard_checkbox = self._server.gui.add_checkbox("摔倒后取消模型控制（走路 / 踢球）", initial_value=True)
      self._server.gui.add_html("倾角 > 60° 或骨盆高度 < 0.30 m，持续 0.1 秒后关闭驱动力矩。触发后选择模式或重置恢复；取消勾选不自动恢复。")

      @guard_checkbox.on_update
      def _(_) -> None:
        self.request_action("CUSTOM", {"fall_guard": bool(guard_checkbox.value)})

    with self._server.gui.add_folder("控制延迟模拟"):
      delay_scope = self._server.gui.add_dropdown("延迟作用范围", options=("全部模型", "仅踢球"), initial_value="全部模型")
      inference_on = self._server.gui.add_checkbox("启用推理延迟", initial_value=False)
      inference_ms = self._server.gui.add_slider("推理延迟 (ms)", 0.0, 200.0, 20.0, 40.0)
      motor_on = self._server.gui.add_checkbox("启用电机延迟", initial_value=False)
      motor_ms = self._server.gui.add_slider("电机延迟 (ms)", 0.0, 200.0, 5.0, 10.0)
      self._delay_html = self._server.gui.add_html(self.controller.latency.status)
      self._server.gui.add_html("推理：延后送出模型结果；电机：延后执行关节目标。使用仿真时间，不阻塞网页。切换模式或修改设置会清空队列，以当前关节位置等待新指令。")

      def update_delay(_) -> None:
        self.request_action("CUSTOM", {"latency": (
          inference_on.value, inference_ms.value, motor_on.value,
          motor_ms.value, delay_scope.value,
        )})

      for handle in (delay_scope, inference_on, inference_ms, motor_on, motor_ms):
        handle.on_update(update_delay)

    self._push = ForcePulse(self.env.unwrapped)
    with self._server.gui.add_folder("推力测试"):
      self._push_direction = self._server.gui.add_dropdown(
        "推力方向（水平随朝向，向下为竖直）", options=tuple(ForcePulse.DIRECTIONS), initial_value="向后",
      )
      self._push_force = self._server.gui.add_slider("推力 (N)", 0.0, 300.0, 5.0, 60.0)
      self._push_duration = self._server.gui.add_slider("持续时间 (s)", 0.02, 1.0, 0.02, 0.2)
      push_btn = self._server.gui.add_button("Push（施加推力）")
      cancel_btn = self._server.gui.add_button("取消推力")
      self._push_html = self._server.gui.add_html("尚未施力；作用于躯干，结束后自动清零。")

      @push_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"push": (
          self._push_direction.value, float(self._push_force.value),
          float(self._push_duration.value), int(self._scene.env_idx),
        )})

      @cancel_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"cancel_push": True})

    self._gamepad = BrowserGamepad(self._server)

  def _execute_step(self) -> bool:
    self._push.before_step()
    success = super()._execute_step()
    if success:
      self._push.after_step()
    else:
      self._push.cancel("仿真异常，推力已清零")
    return success

  def _process_actions(self) -> None:
    if hasattr(self, "_gamepad"):
      twist, buttons = self._gamepad.poll()
      if twist is not None:
        self.controller.twist[:] = torch.tensor(twist, device=self.controller.device)
      for button in buttons:
        if button == "kick":
          self.controller.enter_kick()
        elif button in ("walk", "stop"):
          self.controller.enter_walk()
        elif button == "roll":
          self.controller.enter_roll()
        elif button == "getup":
          self.controller.enter_getup()
        elif button == "ball":
          self.controller.place_ball()
    super()._process_actions()

  def close(self) -> None:
    if hasattr(self, "_push"):
      self._push.cancel()
    if hasattr(self, "_gamepad"):
      self._gamepad.close()
    super().close()

  def _handle_custom_action(self, action, payload) -> bool:
    if isinstance(payload, dict):
      if "latency" in payload:
        self.controller.latency.configure(*payload["latency"])
        return True
      if "fall_guard" in payload:
        self.controller.fall_guard.enabled = payload["fall_guard"]
        self.controller.fall_guard.elapsed.zero_()
        return True
      if "push" in payload:
        self._push.start(*payload["push"])
        return True
      if payload.get("cancel_push"):
        self._push.cancel()
        return True
      if payload.get("roll_demo"):
        self.controller.demo_getup("roll")
        return True
      if payload.get("getup_demo"):
        self.controller.demo_getup()
        return True
      if payload.get("reset_ball"):
        self.controller.place_ball()
        return True
      if "dual_mode" in payload:
        if payload["dual_mode"] == "kick":
          self.controller.enter_kick()
        elif payload["dual_mode"] == "roll":
          self.controller.enter_roll()
        elif payload["dual_mode"] == "getup":
          self.controller.enter_getup()
        else:
          self.controller.enter_walk()
        return True
      if "twist" in payload:
        axis, value = payload["twist"]
        self.controller.twist[:, axis] = value
        return True
    return super()._handle_custom_action(action, payload)

  def reset_environment(self) -> None:
    if hasattr(self, "_push"):
      self._push.cancel()
    super().reset_environment()
    self._vx.value = self._vy.value = self._wz.value = 0.0

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    if self._counter % 4 == 0:
      self._mode_html.content = f"<b>{self.controller._status}</b>"
      self._push_html.content = self._push.status
      self._delay_html.content = self.controller.latency.status


def resolve_checkpoint(value: str) -> Path:
  path = Path(value).expanduser().resolve()
  if path.is_dir():
    if not any(re.fullmatch(r"model_(\d+)\.pt", p.name) for p in path.glob("model_*.pt")):
      runs = sorted((p for p in path.iterdir() if p.is_dir() and any(re.fullmatch(r"model_(\d+)\.pt", f.name) for f in p.glob("model_*.pt"))), reverse=True)
      if not runs:
        raise FileNotFoundError(f"No checkpoints in {path}")
      path = runs[0]
    candidates = [(int(m.group(1)), p) for p in path.glob("model_*.pt")
                  if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not candidates:
      raise FileNotFoundError(f"No model_<iteration>.pt in {path}")
    path = max(candidates)[1]
  if not path.is_file():
    raise FileNotFoundError(path)
  print(f"[INFO] Checkpoint: {path}")
  return path


def smoke_test(env, controller) -> None:
  import importlib.util
  from types import SimpleNamespace

  # Compare against the source project's observation function on identical state.
  source = Path(__file__).resolve().parents[2] / "DroidUpE1_mjlab/src/tasks/amp/mdp/observations.py"
  spec = importlib.util.spec_from_file_location("walk_obs_reference", source)
  assert spec is not None and spec.loader is not None
  reference = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(reference)
  raw = env.unwrapped
  robot = raw.scene["robot"]
  proxy = SimpleNamespace(
    scene=raw.scene, action_manager=raw.action_manager,
    command_manager=SimpleNamespace(get_command=lambda _: controller.twist.expand(raw.num_envs, -1)),
  )
  joint_cfg = SceneEntityCfg("robot", joint_ids=controller.joint_ids)
  torso_cfg = SceneEntityCfg("robot", body_ids=[controller.torso_id])
  with torch.inference_mode():
    # Test the singular supine orientation at a nonzero world heading.
    root = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                      robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w), dim=-1).clone()
    z = torch.zeros(env.num_envs, device=raw.device)
    root[:, 3:7] = quat_from_euler_xyz(z, z, z + 1.1)
    robot.write_root_state_to_sim(root)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    ball_before = raw.scene["ball"].data.root_link_pos_w.clone()
    controller.demo_getup("roll")
    torch.testing.assert_close(raw.scene["ball"].data.root_link_pos_w, ball_before)
    before_q = robot.data.joint_pos.clone()
    before_root = robot.data.root_link_pos_w.clone()
    controller.enter_roll()
    torch.testing.assert_close(robot.data.joint_pos, before_q)
    torch.testing.assert_close(robot.data.root_link_pos_w, before_root)
    assert controller.roll_policy.obs_normalizer._mean.data_ptr() != controller.getup_policy.obs_normalizer._mean.data_ptr()
    roll = raw.command_manager.get_term("roll_motion")
    getup = raw.command_manager.get_term("getup_motion")
    for step in range(roll.motion.time_step_total + getup.motion.time_step_total + 150):
      obs = env.get_observations()
      assert obs["roll_actor"].shape == (env.num_envs, 109)
      torch.testing.assert_close(obs["roll_actor"][:, :40], roll.command)
      torch.testing.assert_close(obs["roll_actor"][:, 46:], obs["actor"][:, 46:])
      action = controller(obs)
      assert torch.isfinite(action).all()
      env.step(action)
      if step % 100 == 0:
        print(f"[SMOKE] roll-chain step={step}, mode={controller.mode}, root_z={robot.data.root_link_pos_w[:, 2].tolist()}, status={controller._status}")
    assert controller.mode == "walk", controller._status
    assert bool((robot.data.root_link_pos_w[:, 2] > 0.45).all())
    print("[SMOKE] roll -> getup -> walk passed")
    env.reset()
    controller.reset()
    for mode, steps in (("stand", 150), ("walk", 200), ("kick", 180), ("walk", 100)):
      if mode == "kick":
        before = robot.data.root_link_pos_w.clone()
        ball_before = raw.scene["ball"].data.root_link_pos_w.clone()
        controller.enter_kick()
        torch.testing.assert_close(ball_before, raw.scene["ball"].data.root_link_pos_w)
        torch.testing.assert_close(before, robot.data.root_link_pos_w)
      elif mode == "walk":
        controller.enter_walk()
        controller.twist[:, 0] = 0.4
        controller.twist[:, 2] = 0.3
      min_z = float("inf")
      for _ in range(steps):
        expected = reference._actor_frame(proxy, joint_cfg, torso_cfg, "twist", corrupt=False)
        torch.testing.assert_close(controller._build_walk_frame(), expected)
        obs = env.get_observations()
        assert obs["actor"].shape == (env.num_envs, 109)
        action = controller(obs)
        assert action.shape == (env.num_envs, 20) and torch.isfinite(action).all()
        env.step(action)
        assert torch.isfinite(robot.data.joint_pos).all()
        min_z = min(min_z, float(robot.data.root_link_pos_w[:, 2].min()))
      assert min_z > 0.35, f"Robot fell during {mode}: root height {min_z:.3f}"
      print(f"[SMOKE] {mode}: {steps} steps, minimum root height={min_z:.3f}")
      if mode == "kick":
        distance = (raw.scene["ball"].data.root_link_pos_w - raw._kick_ball_start_pos).norm(dim=-1)
        print(f"[SMOKE] ball displacement: {distance.tolist()}")
    env.reset()
    controller.reset()
    assert controller.mode == "walk"
    torch.testing.assert_close(controller.history[:, 0], controller.history[:, -1])
    print("[SMOKE] source observation parity, reset and history checks passed")
    # Pure rolling away from the robot: verify actual contact dissipation.
    ball = raw.scene["ball"]
    state = torch.zeros(raw.num_envs, 13, device=raw.device)
    state[:, :3] = raw.scene.env_origins
    state[:, 0] += 5.0
    state[:, 2] += 0.11
    state[:, 3] = 1.0
    state[:, 7] = 2.0
    state[:, 11] = 2.0 / 0.11
    ball.write_root_state_to_sim(state)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    initial = ball.data.root_link_pos_w.clone()
    stopped_at = None
    for step in range(300):
      env.step(controller(env.get_observations()))
      speed = ball.data.root_link_lin_vel_w[:, :2].norm(dim=-1).max().item()
      if speed < 0.03 and stopped_at is None:
        stopped_at = (step + 1) * raw.step_dt
    distance = (ball.data.root_link_pos_w[:, :2] - initial[:, :2]).norm(dim=-1).max().item()
    assert speed < 0.03, f"Ball still rolling: {speed:.3f} m/s"
    assert distance < 5.0, f"Ball travelled too far: {distance:.3f} m"
    print(f"[SMOKE] ball from 2 m/s: stopped at {stopped_at}s, distance={distance:.3f}m, final speed={speed:.5f}m/s")
    # Exercise the complete trained getup clip and the automatic handoff.
    ball_pos = ball.data.root_link_pos_w.clone()
    controller.demo_getup()
    torch.testing.assert_close(ball.data.root_link_pos_w, ball_pos)
    motion = raw.command_manager.get_term("getup_motion")
    kick_frame = raw.command_manager.get_term("motion").time_steps.clone()
    before_q = robot.data.joint_pos.clone()
    before_root = robot.data.root_link_pos_w.clone()
    controller.enter_getup()
    torch.testing.assert_close(robot.data.joint_pos, before_q)
    torch.testing.assert_close(robot.data.root_link_pos_w, before_root)
    assert controller.getup_policy.obs_normalizer._mean.data_ptr() != controller.kick_policy.obs_normalizer._mean.data_ptr()
    for step in range(motion.motion.time_step_total + 100):
      obs = env.get_observations()
      assert obs["getup_actor"].shape == (env.num_envs, 109)
      torch.testing.assert_close(obs["getup_actor"][:, :40], motion.command)
      # The common IMU / joints / previous-action suffix must be identical;
      # only the reference command and orientation differ between these actors.
      torch.testing.assert_close(obs["getup_actor"][:, 46:], obs["actor"][:, 46:])
      action = controller(obs)
      assert torch.isfinite(action).all()
      env.step(action)
      assert torch.isfinite(robot.data.joint_pos).all()
      if step % 200 == 0:
        print(f"[SMOKE] getup step={step}, root_z={robot.data.root_link_pos_w[:, 2].tolist()}, mode={controller.mode}")
    torch.testing.assert_close(raw.command_manager.get_term("motion").time_steps, kick_frame)
    assert controller.mode == "walk", f"Getup failed to finish upright: {controller._status}"
    assert bool((robot.data.root_link_pos_w[:, 2] > 0.45).all())
    print("[SMOKE] getup -> walk passed, independent reference clocks and normalizers verified")
    controller.place_ball()
    controller.enter_kick()
    start_ball = ball.data.root_link_pos_w.clone()
    for _ in range(180):
      env.step(controller(env.get_observations()))
      assert bool((robot.data.root_link_pos_w[:, 2] > 0.35).all())
    ball_travel = (ball.data.root_link_pos_w[:, :2] - start_ball[:, :2]).norm(dim=-1)
    controller.enter_walk()
    for _ in range(100):
      env.step(controller(env.get_observations()))
      assert bool((robot.data.root_link_pos_w[:, 2] > 0.45).all())
    print(f"[SMOKE] getup -> walk -> kick -> walk passed; kick ball travel={ball_travel.tolist()}")



def push_smoke_test(env, controller):
  raw = env.unwrapped
  robot = raw.scene["robot"]
  push = ForcePulse(raw)
  with torch.inference_mode():
    for _ in range(100):
      env.step(controller(env.get_observations()))
    initial_velocity = robot.data.root_link_lin_vel_w.clone()
    push.start("向后", 80., 0.2)
    steps = push.remaining
    for _ in range(steps):
      push.before_step()
      torch.testing.assert_close(robot.data.body_external_force[:, push.body_id:push.body_id+1], push.force)
      env.step(controller(env.get_observations()))
      push.after_step()
    assert push.remaining == 0
    assert not robot.data.body_external_wrench[:, push.body_id].any()
    delta = (robot.data.root_link_lin_vel_w - initial_velocity).norm(dim=-1).max().item()
    assert delta > 0.05, f"Push produced no measurable velocity change: {delta}"
    push.start("向左", 100., 1.)
    push.before_step()
    push.cancel()
    assert not robot.data.body_external_wrench[:, push.body_id].any()
    assert torch.isfinite(robot.data.joint_pos).all()
    print(f"[PUSH TEST] 80 N for {steps * raw.step_dt:.2f}s; velocity change={delta:.3f} m/s; expiry and cancel clear wrench")


def run(cfg: WalkKickPlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  walk_path = resolve_checkpoint(cfg.walk_checkpoint)
  kick_path = resolve_checkpoint(cfg.kick_checkpoint)
  getup_path = resolve_checkpoint(cfg.getup_checkpoint)
  roll_path = resolve_checkpoint(cfg.roll_checkpoint)
  if not walk_path.is_file():
    raise FileNotFoundError(walk_path)
  if not kick_path.is_file():
    raise FileNotFoundError(kick_path)

  env_cfg = load_env_cfg(cfg.kick_task, play=True)
  agent_cfg = load_rl_cfg(cfg.kick_task)
  walk_cfg = load_env_cfg(cfg.walk_task, play=True)
  getup_cfg = load_env_cfg(cfg.getup_task, play=True)
  roll_cfg = load_env_cfg(cfg.roll_task, play=True)
  if abs(roll_cfg.decimation * roll_cfg.sim.mujoco.timestep - env_cfg.decimation * env_cfg.sim.mujoco.timestep) > 1e-9:
    raise ValueError("Roll must use the same policy control period")
  if getup_cfg.decimation != env_cfg.decimation or getup_cfg.sim.mujoco.timestep != env_cfg.sim.mujoco.timestep:
    raise ValueError("Getup and kick must use the same control timestep")
  # Keep getup MJCF (wider hip limits) but overlay AMP walk actuators (Booster PD).
  robot = deepcopy(getup_cfg.scene.entities["robot"])
  walk_robot = walk_cfg.scene.entities["robot"]
  robot.articulation = deepcopy(walk_robot.articulation)
  env_cfg.scene.entities["robot"] = robot
  print(
    "[INFO] Robot: getup MJCF + walk AMP actuators "
    f"(task={cfg.walk_task}); kick/getup keep legacy init_state"
  )
  walk_align = _load_walk_deployment_alignment(walk_path)
  walk_scale = ((walk_align.get("actions") or {}).get("joint_pos") or {}).get("scale")
  if walk_scale is None:
    walk_scale = walk_cfg.actions["joint_pos"].scale
    print("[INFO] Walk action scale from task cfg (no deployment_alignment scale)")
  else:
    print(
      f"[INFO] Walk action scale from {walk_path.parent.name}/deployment_alignment.yaml"
    )
  walk_home = _walk_home_from_alignment(walk_align) or dict(
    walk_robot.init_state.joint_pos
  )
  env_cfg.commands["getup_motion"] = deepcopy(getup_cfg.commands["motion"])
  env_cfg.commands["getup_motion"].debug_vis = False
  getup_obs = deepcopy(getup_cfg.observations["actor"])
  for term in getup_obs.terms.values():
    if term.params.get("command_name") == "motion":
      term.params["command_name"] = "getup_motion"
  getup_obs.terms["motion_anchor_ori_b"].func = aligned_motion_orientation
  env_cfg.observations["getup_actor"] = getup_obs
  # Use the shared reference implementation (name-based body mapping, no RSI).
  roll_cmd = deepcopy(env_cfg.commands["getup_motion"])
  roll_cmd.motion_file = roll_cfg.commands["motion"].motion_file
  roll_cmd.body_names = roll_cfg.commands["motion"].body_names
  roll_cmd.anchor_body_name = roll_cfg.commands["motion"].anchor_body_name
  env_cfg.commands["roll_motion"] = roll_cmd
  roll_obs = deepcopy(roll_cfg.observations["actor"])
  for term in roll_obs.terms.values():
    if term.params.get("command_name") == "motion":
      term.params["command_name"] = "roll_motion"
  roll_obs.terms["motion_anchor_ori_b"].func = aligned_motion_orientation
  env_cfg.observations["roll_actor"] = roll_obs
  env_cfg.scene.num_envs = cfg.num_envs
  if not 0.0 <= cfg.ball_rolling_friction <= 0.1:
    raise ValueError("ball_rolling_friction must be between 0 and 0.1 m")
  # Sliding friction alone cannot stop a sphere in pure rolling. Enable the
  # two rolling-contact torque directions and let this ball select its friction.
  env_cfg.scene.entities["ball"].collisions = (CollisionCfg(
    geom_names_expr=("ball_geom",), condim=6, priority=1,
    friction=(0.7, 0.01, cfg.ball_rolling_friction),
  ),)
  env_cfg.rewards = {}
  env_cfg.observations["actor"].terms["motion_anchor_ori_b"].func = aligned_motion_orientation
  if "motion" in env_cfg.commands:
    env_cfg.commands["motion"].debug_vis = False
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] Terminations disabled")
  # Kick play still keeps COM / encoder / friction DR — strip for walk stability.
  for _k in ("base_com", "encoder_bias", "foot_friction"):
    env_cfg.events.pop(_k, None)
  # Match DroidUp walk contact solver budget (kick defaults are tighter).
  env_cfg.sim.nconmax = max(int(getattr(env_cfg.sim, "nconmax", 0) or 0), getup_cfg.sim.nconmax)
  env_cfg.sim.njmax = max(int(getattr(env_cfg.sim, "njmax", 0) or 0), getup_cfg.sim.njmax)

  env_cfg.events.pop("ball_mass", None)
  env_cfg.events.pop("ball_friction", None)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  disable_motion_rsi(env)
  disable_motion_rsi(env, "getup_motion", "getup")
  disable_motion_rsi(env, "roll_motion", "roll")
  snap_robot_to_walk_play_pose(env)
  place_ball_only(env)
  print("[INFO] Start: home pose + ball nearby; walk actor active")

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.kick_task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(kick_path), map_location=device)
  kick_policy = runner.get_inference_policy(device=device)
  print(f"[INFO] Loaded kick policy from {kick_path.name}")

  # Clone the architecture only, then strictly replace every parameter and
  # normalization buffer. The kick and getup normalizers remain independent.
  getup_policy = deepcopy(kick_policy)
  getup_state = torch.load(getup_path, map_location=device, weights_only=False)["actor_state_dict"]
  getup_policy.load_state_dict(getup_state, strict=True)
  getup_policy.eval()
  print(f"[INFO] Loaded getup policy from {getup_path}")
  roll_policy = deepcopy(kick_policy)
  roll_state = torch.load(roll_path, map_location=device, weights_only=False)["actor_state_dict"]
  roll_policy.load_state_dict(roll_state, strict=True)
  roll_policy.eval()
  print(f"[INFO] Loaded roll policy from {roll_path}")
  walk_actor = _load_walk_actor(walk_path, device)
  controller = MultiPolicyController(
    env, kick_policy, walk_actor, device, getup_policy,
    getup_cfg.actions["joint_pos"].scale, roll_policy,
    roll_cfg.actions["joint_pos"].scale, walk_scale, walk_home, cfg.stand_hold,
  )
  controller.reset()
  print("[INFO] Walk obs=345 (5 x 69, no normalization); kick/getup/roll obs=109 each (independent checkpoint normalization)")
  if cfg.smoke_test or cfg.push_smoke_test:
    try:
      if cfg.push_smoke_test:
        push_smoke_test(env, controller)
      else:
        smoke_test(env, controller)
    finally:
      env.close()
    return
  controller.enter_walk()
  print("[INFO] Policy mode = WALK (auto)")

  viser_server = viser.ViserServer(
    host=cfg.viser_host, port=int(cfg.viser_port), label="walk + kick + roll + getup"
  )
  print(f"[INFO] Viser http://127.0.0.1:{cfg.viser_port}/")
  viewer = WalkKickViewer(env, controller, viser_server=viser_server)
  viewer.controller = controller
  # Ensure walk before the sim loop starts.
  controller.enter_walk()
  try:
    viewer.run()
  finally:
    env.close()
    viser_server.stop()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  workspace = Path(__file__).resolve().parents[2]
  default_walk = workspace / "DroidUpE1_mjlab/logs/rsl_rl/wbot_20dof_walk_run_amp/2026-09-24_08-55-34"
  default_kick = workspace / "unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_kick/2026-09-24_16-56-09_ball_perenv_pelvis_exclude"

  default_getup = workspace / "unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_getup_tracking/2026-09-28_15-38-09_wbot_20dof_roban_no_state_estimation/model_30000.pt"

  default_roll = workspace / "unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_roll_to_prone_tracking"
  # Roll runs are empty in this workspace; reuse getup weights so Walk/Kick still launch.
  # Replace with a real roll checkpoint when available (--roll-checkpoint).
  try:
    resolve_checkpoint(str(default_roll))
  except FileNotFoundError:
    default_roll = default_getup
    print(
      "[WARN] No roll checkpoints under wbot_20dof_roll_to_prone_tracking; "
      "using getup checkpoint as roll stub (Roll button is not a real roll policy)."
    )

  cfg = tyro.cli(
    WalkKickPlayConfig,
    default=WalkKickPlayConfig(
      walk_checkpoint=str(default_walk),
      kick_checkpoint=str(default_kick),
      getup_checkpoint=str(default_getup),
      roll_checkpoint=str(default_roll),
    ),
  )
  run(cfg)


if __name__ == "__main__":
  root = Path(__file__).resolve().parents[1]
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  os.chdir(root)
  main()
