"""Viser play viewer that surfaces AMP style reward on the Rewards bar."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
import viser

from mjlab.viewer import ViserPlayViewer
from rsl_rl.algorithms import AmpPPO

from src.tasks.amp.rl.ankle_roll_forward import (
  ANKLE_ROLL_FORWARD_MAX_M,
  AnkleRollForwardController,
)
from src.tasks.amp.rl.sole_height import SOLE_HEIGHT_MAX_M, SoleHeightController


class AmpViserPlayViewer(ViserPlayViewer):
  """``ViserPlayViewer`` that prepends live ``style_reward`` to the reward bar."""

  STYLE_TERM = "style_reward"

  def __init__(
    self,
    env,
    policy,
    *,
    amp_algorithm: AmpPPO | None = None,
    sole_controller: SoleHeightController | None = None,
    ankle_roll_forward_controller: AnkleRollForwardController | None = None,
    **kwargs,
  ) -> None:
    super().__init__(env, policy, **kwargs)
    self._amp_alg = amp_algorithm
    self._sole_controller = sole_controller
    self._ankle_roll_forward_controller = ankle_roll_forward_controller
    self._prev_amp: torch.Tensor | None = None
    self._style_reward = torch.zeros(env.num_envs, device=env.unwrapped.device)
    self._last_style_step: int | None = None

  def setup(self) -> None:
    gui = self._server.gui
    orig_add_tab_group = gui.add_tab_group
    held: dict[str, Any] = {}

    def _capture_tab_group(*args, **kwargs):
      tabs = orig_add_tab_group(*args, **kwargs)
      held["tabs"] = tabs
      return tabs

    gui.add_tab_group = _capture_tab_group  # type: ignore[method-assign]
    try:
      super().setup()
    finally:
      gui.add_tab_group = orig_add_tab_group  # type: ignore[method-assign]

    self._setup_sole_height_gui(held.get("tabs"))
    self._setup_ankle_roll_forward_gui(held.get("tabs"))
    if self._amp_alg is None or self._term_overlays is None:
      return
    self._inject_style_reward_term()
    original_update = self._term_overlays.update

    def update_with_style(paused: bool) -> None:
      if not paused:
        self._refresh_style_reward()
      original_update(paused)
      if paused or self._term_overlays is None:
        return
      env_idx = self._scene.env_idx
      value = float(self._style_reward[env_idx].detach().cpu())
      arr = np.asarray([value], dtype=np.float64)
      panel = self._term_overlays.reward_bar_panel
      if panel is not None:
        panel.update([(self.STYLE_TERM, arr)])
      plotter = self._term_overlays.reward_plotter
      if plotter is not None and self.STYLE_TERM in plotter._terms:
        plotter.update([(self.STYLE_TERM, arr)])

    self._term_overlays.update = update_with_style  # type: ignore[method-assign]
    print("[INFO] AMP style_reward added to Rewards bar.")

  def _setup_sole_height_gui(self, tabs: Any | None) -> None:
    ctrl = self._sole_controller
    if ctrl is None:
      return

    def _build_controls() -> None:
      slider = self._server.gui.add_slider(
        "Height (m)",
        min=0.0,
        max=SOLE_HEIGHT_MAX_M,
        step=0.001,
        initial_value=ctrl.height_m,
      )
      label = self._server.gui.add_markdown(
        f"**{ctrl.height_m * 1000.0:.0f} mm** "
        f"(max {SOLE_HEIGHT_MAX_M * 100.0:.0f} cm)"
      )

      @slider.on_update
      def _on_sole_height(event) -> None:
        h = ctrl.set(float(event.target.value))
        label.content = (
          f"**{h * 1000.0:.0f} mm** "
          f"(max {SOLE_HEIGHT_MAX_M * 100.0:.0f} cm)"
        )

    if tabs is not None:
      with tabs.add_tab("Sole", icon=viser.Icon.SETTINGS):
        _build_controls()
    else:
      with self._server.gui.add_folder("Sole"):
        _build_controls()

    print(
      f"[INFO] Viser Sole slider ready (initial {ctrl.height_m * 1000.0:.0f} mm).",
      flush=True,
    )

  def _setup_ankle_roll_forward_gui(self, tabs: Any | None) -> None:
    ctrl = self._ankle_roll_forward_controller
    if ctrl is None:
      return

    def _build_controls() -> None:
      slider = self._server.gui.add_slider(
        "Forward (m)",
        min=0.0,
        max=ANKLE_ROLL_FORWARD_MAX_M,
        step=0.001,
        initial_value=ctrl.offset_m,
      )
      label = self._server.gui.add_markdown(
        f"**{ctrl.offset_m * 1000.0:.0f} mm** "
        f"(max {ANKLE_ROLL_FORWARD_MAX_M * 1000.0:.0f} mm)"
      )

      @slider.on_update
      def _on_ankle_roll_forward(event) -> None:
        dx = ctrl.set(float(event.target.value))
        label.content = (
          f"**{dx * 1000.0:.0f} mm** "
          f"(max {ANKLE_ROLL_FORWARD_MAX_M * 1000.0:.0f} mm)"
        )

    if tabs is not None:
      with tabs.add_tab("AnkleX", icon=viser.Icon.ADJUSTMENTS):
        _build_controls()
    else:
      with self._server.gui.add_folder("AnkleRollForward"):
        _build_controls()

    print(
      f"[INFO] Viser AnkleX slider ready "
      f"(initial {ctrl.offset_m * 1000.0:.0f} mm).",
      flush=True,
    )

  def _inject_style_reward_term(self) -> None:
    overlays = self._term_overlays
    assert overlays is not None
    name = self.STYLE_TERM
    panel = overlays.reward_bar_panel
    if panel is not None and name not in panel._term_names:
      # Keep style_reward first so it survives RewardBarPanel.max_terms=20.
      panel._term_names = [name, *panel._term_names][:20]
      panel._histories[name] = deque(maxlen=panel._window_steps)
      keep = set(panel._term_names)
      for key in list(panel._histories):
        if key not in keep:
          del panel._histories[key]

    plotter = overlays.reward_plotter
    if plotter is not None and name not in plotter._terms:
      from mjlab.viewer.viser.term_plotter import _TermState, _color_for

      state = _TermState(
        name=name,
        color=_color_for(0),
        history=deque(maxlen=plotter._history_length),
        enabled=True,
      )
      plotter._terms = {name: state, **plotter._terms}
      if hasattr(plotter, "_checkboxes"):
        cb = plotter._server.gui.add_checkbox(f"[AMP] {name}", initial_value=True)

        @cb.on_update
        def _(event, term=name) -> None:
          st = plotter._terms[term]
          st.enabled = bool(event.target.value)
          if st.enabled and st.plot is None:
            plotter._create_plot(st)
          elif not st.enabled and st.plot is not None:
            st.plot.remove()
            st.plot = None

        plotter._checkboxes[name] = cb
        plotter._create_plot(state)

  def _refresh_style_reward(self) -> None:
    alg = self._amp_alg
    if alg is None:
      return
    if alg._env is None:
      alg._env = self.env

    env = self.env.unwrapped
    step = int(getattr(env, "common_step_counter", -1))
    if self._last_style_step is not None and step == self._last_style_step:
      return

    obs = self.env.get_observations()
    if "amp" not in obs.keys():
      return
    amp = obs["amp"].detach()
    if self._prev_amp is None or self._prev_amp.shape != amp.shape:
      self._prev_amp = amp.clone()
      self._style_reward = torch.zeros(amp.shape[0], device=amp.device)
      self._last_style_step = step
      return

    command = env.command_manager.get_command("twist")
    if command is None:
      return
    with torch.no_grad():
      label = alg._command_motion_label(command)
      state_norm = torch.cat(
        (alg.amp_normalizer.normalize(self._prev_amp), label), dim=1
      )
      next_norm = torch.cat((alg.amp_normalizer.normalize(amp), label), dim=1)
      style, _ = alg.discriminator.style_reward(
        state_norm, next_norm, alg.amp_reward_coefficient
      )
      self._style_reward = style
    self._prev_amp = amp.clone()
    self._last_style_step = step

  def reset_environment(self) -> None:  # type: ignore[override]
    super().reset_environment()
    self._prev_amp = None
    self._last_style_step = None
    self._style_reward = torch.zeros(
      self.env.num_envs, device=self.env.unwrapped.device
    )
