"""Viser play viewer for Kick: motion clip controls + torso push button."""

from __future__ import annotations

from typing import Any, Optional

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction

from src.viewer.force_pulse import ForcePulse


class KickViserPlayViewer(ViserPlayViewer):
  """``ViserPlayViewer`` with a torso push panel (same as walk-kick play)."""

  def setup(self) -> None:
    super().setup()
    self._push = ForcePulse(self.env.unwrapped)
    with self._server.gui.add_folder("推力"):
      self._push_direction = self._server.gui.add_dropdown(
        "方向（水平随朝向）",
        options=tuple(ForcePulse.DIRECTIONS),
        initial_value="向后",
      )
      self._push_force = self._server.gui.add_slider(
        "推力 (N)", min=0.0, max=300.0, step=5.0, initial_value=60.0
      )
      self._push_duration = self._server.gui.add_slider(
        "持续时间 (s)", min=0.02, max=1.0, step=0.02, initial_value=0.2
      )
      push_btn = self._server.gui.add_button("Push")
      cancel_btn = self._server.gui.add_button("取消")
      self._push_html = self._server.gui.add_html(
        "尚未施力；作用于 torso_link，结束后自动清零。"
      )

      @push_btn.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM",
          {
            "push": (
              self._push_direction.value,
              float(self._push_force.value),
              float(self._push_duration.value),
              int(self._scene.env_idx),
            )
          },
        )

      @cancel_btn.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"cancel_push": True})

    print("[INFO] Kick push panel ready (torso ForcePulse).", flush=True)

  def _execute_step(self) -> bool:
    self._push.before_step()
    success = super()._execute_step()
    if success:
      self._push.after_step()
    else:
      self._push.cancel("仿真异常，推力已清零")
    if hasattr(self, "_push_html"):
      self._push_html.content = self._push.status
    return success

  def _handle_custom_action(
    self,
    action: ViewerAction,
    payload: Optional[Any],
  ) -> bool:
    if isinstance(payload, dict):
      if "push" in payload:
        self._push.start(*payload["push"])
        self._push_html.content = self._push.status
        return True
      if payload.get("cancel_push"):
        self._push.cancel()
        self._push_html.content = self._push.status
        return True
    return super()._handle_custom_action(action, payload)

  def close(self) -> None:
    if hasattr(self, "_push"):
      self._push.cancel()
    super().close()
