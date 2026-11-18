"""Motion command that places the ball after the robot is reset to the clip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
import viser

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

from src.tasks.kick.mdp.events import place_ball_at_world, place_ball_in_front_of_right_foot

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class KickMotionCommand(MotionCommand):
  """Motion imitation command; places the kick ball after writing robot state."""

  cfg: KickMotionCommandCfg

  def __init__(self, cfg: KickMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # When ``loop=False``: hold reference at ``_hold_frame`` until Play Clip.
    self._clip_playing = False
    self._hold_frame = 0
    # SE(2)-ish align: motion hold-frame (or clip-start) anchor → robot (xy + yaw).
    self._align_on = False
    self._align_yaw = torch.zeros(self.num_envs, 4, device=self.device)
    self._align_yaw[:, 0] = 1.0
    self._align_t = torch.zeros(self.num_envs, 3, device=self.device)
    self._align_f0 = torch.zeros(self.num_envs, 3, device=self.device)

  def _place_ball_after_kinematics(self, env_ids: torch.Tensor) -> None:
    if self.cfg.ball_placement in ("none", "off", ""):
      return
    # write_*_to_sim does not refresh body_link_pos_w until forward().
    self._env.scene.write_data_to_sim()
    self._env.sim.forward()
    if self.cfg.ball_placement == "world":
      place_ball_at_world(
        self._env,
        env_ids,
        ball_name=self.cfg.ball_name,
        robot_name=self.cfg.entity_name,
        pos=self.cfg.ball_pos,
        pos_noise=self.cfg.ball_pos_noise,
      )
    else:
      place_ball_in_front_of_right_foot(
        self._env,
        env_ids,
        robot_cfg=SceneEntityCfg(self.cfg.entity_name),
        ball_name=self.cfg.ball_name,
        foot_body_name=self.cfg.ball_foot_body_name,
        forward_range=self.cfg.ball_forward_range,
        lateral_range=self.cfg.ball_lateral_range,
        height=float(self.cfg.ball_height),
      )

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self._place_ball_after_kinematics(env_ids)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    super().reset_to_frame(env_ids, frame)
    self._place_ball_after_kinematics(env_ids)

  def _capture_align_to_robot(self, frame: int = 0) -> None:
    """Map motion ``frame`` anchor onto the robot (xy + yaw; keep motion height)."""
    t = int(max(0, min(frame, self.motion.time_step_total - 1)))
    f0 = (
      self.motion.body_pos_w[t, self.motion_anchor_body_index]
      .unsqueeze(0)
      .expand(self.num_envs, -1)
      + self._env.scene.env_origins
    )
    f0_quat = (
      self.motion.body_quat_w[t, self.motion_anchor_body_index]
      .unsqueeze(0)
      .expand(self.num_envs, -1)
    )
    robot_pos = self.robot_anchor_pos_w
    robot_quat = self.robot_anchor_quat_w
    self._align_yaw = yaw_quat(quat_mul(robot_quat, quat_inv(f0_quat)))
    self._align_t = robot_pos.clone()
    self._align_t[:, 2] = f0[:, 2]
    self._align_f0 = f0.clone()
    self._align_on = True

  def _broadcast_align(self, ref: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcast (N, *) align tensors to ``ref``'s leading dims (N, ..., *)."""
    f0, yaw, t = self._align_f0, self._align_yaw, self._align_t
    # Insert singleton dims between batch and last feature dim, then expand.
    while f0.ndim < ref.ndim:
      f0 = f0.unsqueeze(1)
      yaw = yaw.unsqueeze(1)
      t = t.unsqueeze(1)
    expand_shape = ref.shape[:-1] + f0.shape[-1:]
    yaw_shape = ref.shape[:-1] + yaw.shape[-1:]
    t_shape = ref.shape[:-1] + t.shape[-1:]
    return f0.expand(expand_shape), yaw.expand(yaw_shape), t.expand(t_shape)

  def _xform_pos(self, pos: torch.Tensor) -> torch.Tensor:
    if not self._align_on:
      return pos
    f0, yaw, t = self._broadcast_align(pos)
    return t + quat_apply(yaw, pos - f0)

  def _xform_quat(self, quat: torch.Tensor) -> torch.Tensor:
    if not self._align_on:
      return quat
    _, yaw, _ = self._broadcast_align(quat)
    return quat_mul(yaw, quat)

  def _xform_vec(self, vec: torch.Tensor) -> torch.Tensor:
    if not self._align_on:
      return vec
    _, yaw, _ = self._broadcast_align(vec)
    return quat_apply(yaw, vec)

  @property
  def body_pos_w(self) -> torch.Tensor:
    raw = (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )
    return self._xform_pos(raw)

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._xform_quat(self.motion.body_quat_w[self.time_steps])

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._xform_vec(self.motion.body_lin_vel_w[self.time_steps])

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._xform_vec(self.motion.body_ang_vel_w[self.time_steps])

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    raw = (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )
    return self._xform_pos(raw)

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._xform_quat(
      self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
    )

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._xform_vec(
      self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._xform_vec(
      self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
    )

  def _clamp_hold_frame(self, frame: int) -> int:
    return int(max(0, min(frame, self.motion.time_step_total - 1)))

  def _hold_at_selected_frame(self) -> None:
    """Pin motion *command* at ``_hold_frame`` and keep it glued to the robot."""
    f = self._clamp_hold_frame(self._hold_frame)
    self._hold_frame = f
    self.time_steps.fill_(f)
    self._capture_align_to_robot(frame=f)
    if hasattr(self, "_scrubber_handles"):
      self._scrubber_handles[0].value = f
    hold_input = getattr(self, "_hold_frame_input", None)
    if hold_input is not None and int(hold_input.value) != f:
      hold_input.value = f
    self.update_relative_body_poses()

  def _update_command(self):
    if self.cfg.loop:
      super()._update_command()
      return

    # Play-once reference: pin model input at Hold Frame until Play Clip.
    if not self._clip_playing:
      self._hold_at_selected_frame()
      return

    self.time_steps += 1
    done = self.time_steps >= self.motion.time_step_total
    if torch.any(done):
      # Clip finished: pin input at Hold Frame again. Do NOT reset the env.
      self._clip_playing = False
      self._hold_at_selected_frame()
      self._set_play_clip_disabled(False)
      self._set_hold_frame_disabled(False)
      status = getattr(self, "_clip_status", None)
      if status is not None:
        status.content = (
          f"**Holding frame {self._hold_frame}** (aligned to robot)"
        )
      return

    self.update_relative_body_poses()

  def start_play_clip(self) -> bool:
    """Same as Viser Play Clip: freeze align at frame 0, then advance once."""
    if self.cfg.loop or self._clip_playing:
      return False
    self.time_steps.zero_()
    self._capture_align_to_robot(frame=0)
    self._clip_playing = True
    self._set_play_clip_disabled(True)
    self._set_hold_frame_disabled(True)
    status = getattr(self, "_clip_status", None)
    if status is not None:
      status.content = "**Playing clip…** (aligned at start)"
    return True

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    super().create_gui(name, server, get_env_idx, on_change, request_action)
    if self.cfg.loop:
      return

    # Ensure ghost/frames can render during play-once.
    self.cfg.debug_vis = True

    max_frame = int(self.motion.time_step_total) - 1
    with server.gui.add_folder("Motion Clip"):
      self._play_clip_btn = server.gui.add_button(
        "Play Clip",
        icon=viser.Icon.PLAYER_PLAY,
      )
      self._hold_frame_input = server.gui.add_number(
        "Hold Frame",
        initial_value=int(self._hold_frame),
        min=0,
        max=max_frame,
        step=1,
      )
      self._clip_status = server.gui.add_markdown(
        f"**Holding frame {self._hold_frame}** (aligned to robot)"
      )

      show_cb = server.gui.add_checkbox(
        "Show motion",
        initial_value=bool(self._debug_vis_enabled),
      )
      mode_dd = server.gui.add_dropdown(
        "Vis mode",
        options=["ghost", "frames"],
        initial_value=str(self.cfg.viz.mode),
      )

      @self._hold_frame_input.on_update
      def _on_hold_frame(event) -> None:
        if self._clip_playing:
          return
        self._hold_frame = self._clamp_hold_frame(int(event.target.value))
        self._hold_at_selected_frame()
        status = getattr(self, "_clip_status", None)
        if status is not None:
          status.content = (
            f"**Holding frame {self._hold_frame}** (aligned to robot)"
          )
        if on_change is not None:
          on_change()

      @show_cb.on_update
      def _on_show(event) -> None:
        self._debug_vis_enabled = bool(event.target.value)
        if on_change is not None:
          on_change()

      @mode_dd.on_update
      def _on_mode(event) -> None:
        mode = str(event.target.value)
        if mode in ("ghost", "frames"):
          self.cfg.viz.mode = mode  # type: ignore[assignment]
          # Force rebuild ghost mesh if switching back to ghost.
          if mode == "ghost":
            self._ghost_model = None
          if on_change is not None:
            on_change()

      @self._play_clip_btn.on_click
      def _(_) -> None:
        if self.start_play_clip() and on_change is not None:
          on_change()

    self._set_scrubber_disabled(False)
    print(
      "[INFO] Motion play-once: Hold Frame follows robot (xy+yaw); "
      "Play Clip freezes align at frame 0 and advances once; "
      "when done, returns to Hold Frame. "
      "Toggle 'Show motion' for ghost/frames.",
      flush=True,
    )

  def _set_play_clip_disabled(self, disabled: bool) -> None:
    btn = getattr(self, "_play_clip_btn", None)
    if btn is not None:
      btn.disabled = disabled

  def _set_hold_frame_disabled(self, disabled: bool) -> None:
    hold_input = getattr(self, "_hold_frame_input", None)
    if hold_input is not None:
      hold_input.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if self.cfg.loop:
      super().on_viewer_pause(paused)
      return
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(self._clip_playing)
    self._set_hold_frame_disabled(self._clip_playing)


@dataclass(kw_only=True)
class KickMotionCommandCfg(MotionCommandCfg):
  ball_name: str = "ball"
  ball_foot_body_name: str = "right_ankle_roll_link"
  ball_height: float = 0.11
  # Body frame: +x forward, +y left. 右前方 = forward + right (−y).
  ball_forward_range: tuple[float, float] = (0.32, 0.40)
  ball_lateral_range: tuple[float, float] = (-0.10, -0.05)
  # Fixed world placement (tuned in Viser) with small domain randomisation.
  ball_placement: str = "world"  # "world" | "foot" | "none"
  ball_pos: tuple[float, float, float] = (0.095, 0.621, 0.110)
  ball_pos_noise: tuple[float, float, float] = (0.02, 0.02, 0.01)
  # True: wrap and keep advancing. False (play): pin at Hold Frame until Play Clip.
  loop: bool = True
  # Present on newer mjlab MotionCommandCfg; keep here so older installs still load.
  start_time_step: int = 0
  start_time_step_prob: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> KickMotionCommand:
    return KickMotionCommand(self, env)
