"""Native MuJoCo play viewer for Kick: same Motion Clip + push logic as Viser."""

from __future__ import annotations

from typing import Any, Optional

from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_B, KEY_C, KEY_G, KEY_H, KEY_T, KEY_U, KEY_Y

from src.tasks.kick.mdp.commands import KickMotionCommand
from src.viewer.force_pulse import ForcePulse
from src.viewer.stable_camera_viewer import StableCameraNativeViewer

_PUSH_KEYS = {
  KEY_T: "向后",
  KEY_Y: "向前",
  KEY_G: "向左",
  KEY_H: "向右",
  KEY_B: "向下",
}


class KickNativePlayViewer(StableCameraNativeViewer):
  """Native viewer with Viser-equivalent kick controls.

  Keys (in addition to default native viewer keys)::

    C  Play Clip（对齐当前机器人后播一次参考；play 时 loop=False）
    T  向后推 torso（默认 60 N / 0.2 s）
    Y  向前推
    G  向左推
    H  向右推
    B  向下推
    U  取消推力
  """

  def setup(self) -> None:
    super().setup()
    self._push = ForcePulse(self.env.unwrapped)
    self._push_force = 60.0
    self._push_duration = 0.2
    print(
      "[INFO] Kick native play: C=Play Clip | T/Y/G/H/B=push torso | U=cancel push",
      flush=True,
    )

  def _safe_key_callback(self, key: int) -> None:
    # Queue only; do not touch env on the viewer thread.
    if key == KEY_C:
      self.request_action("CUSTOM", {"play_clip": True})
      return
    if key == KEY_U:
      self.request_action("CUSTOM", {"cancel_push": True})
      return
    direction = _PUSH_KEYS.get(key)
    if direction is not None:
      self.request_action(
        "CUSTOM",
        {
          "push": (
            direction,
            float(self._push_force),
            float(self._push_duration),
            int(self.env_idx),
          )
        },
      )
      return
    super()._safe_key_callback(key)

  def _execute_step(self) -> bool:
    self._push.before_step()
    success = super()._execute_step()
    if success:
      self._push.after_step()
    else:
      self._push.cancel("仿真异常，推力已清零")
    return success

  def _handle_custom_action(
    self,
    action: ViewerAction,
    payload: Optional[Any],
  ) -> bool:
    if isinstance(payload, dict):
      if payload.get("play_clip"):
        self._start_play_clip()
        return True
      if "push" in payload:
        self._push.start(*payload["push"])
        print(f"[INFO] {self._push.status}", flush=True)
        return True
      if payload.get("cancel_push"):
        self._push.cancel()
        print(f"[INFO] {self._push.status}", flush=True)
        return True
    return super()._handle_custom_action(action, payload)

  def _start_play_clip(self) -> None:
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    if not isinstance(cmd, KickMotionCommand):
      print("[WARN] Play Clip: motion term is not KickMotionCommand", flush=True)
      return
    if cmd.cfg.loop:
      print("[WARN] Play Clip ignored (motion.loop=True)", flush=True)
      return
    if cmd.start_play_clip():
      print("[INFO] Playing clip… (aligned at start)", flush=True)
    else:
      print("[INFO] Clip already playing", flush=True)

  def close(self) -> None:
    if hasattr(self, "_push"):
      self._push.cancel()
    super().close()
