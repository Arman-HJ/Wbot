"""Browser controls over the existing Viser websocket; no extra listening port."""
from __future__ import annotations

from collections import deque
import html
import json
import math
from pathlib import Path
import threading
import time
import uuid

import viser
from viser import _messages


class BrowserGamepad:
  TIMEOUT = 0.4

  def __init__(self, server):
    self.server = server
    self.token = str(uuid.uuid4())
    self._lock = threading.Lock()
    self._owner = None
    self._seen = 0.0
    self._seq = -1
    self._axes = (0.0, 0.0, 0.0)
    self._buttons = set()
    self._events = deque(maxlen=16)
    self._active = False
    self._stop_pending = False
    # Isolate Viser's internal transport API here. The UUID is deliberately not
    # a GUI input: the normal GUI handler ignores it, and this handler consumes it.
    server._websock_server.register_handler(_messages.GuiUpdateMessage, self._receive)
    source = Path(__file__).with_name("gamepad.html").read_text()
    source = source.replace("__CONTROL_UUID__", json.dumps(self.token))
    source = source.replace("__VISER_PROTOCOL__", json.dumps("viser-v" + viser.__version__))
    with server.gui.add_folder("手柄 / 网页摇杆"):
      server.gui.add_html(
        '<iframe title="机器人手柄控制" allow="gamepad" '
        'style="width:100%;height:480px;border:0;border-radius:8px" srcdoc="'
        + html.escape(source, quote=True) + '"></iframe>'
      )

  async def _receive(self, client_id, message):
    if message.uuid != self.token:
      return
    raw = message.updates.get("value")
    if not isinstance(raw, str) or len(raw) > 2048:
      return
    try:
      data = json.loads(raw)
      axes = data["axes"]
      if len(axes) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in axes):
        return
      axes = tuple(max(-1.0, min(1.0, x)) for x in axes)
      seq = data["seq"]
      enabled = data["enabled"]
      if type(seq) is not int or type(enabled) is not bool:
        return
      buttons = set(data["buttons"])
      if not buttons <= {"walk", "kick", "stop", "ball", "getup", "roll"}:
        return
    except (ValueError, TypeError, KeyError):
      return
    now = time.monotonic()
    with self._lock:
      if self._owner is not None and now - self._seen > self.TIMEOUT:
        self._release()
      if self._owner not in (None, client_id):
        return
      if not enabled:
        if self._owner == client_id:
          self._release()
        return
      if self._owner == client_id and seq <= self._seq:
        return
      self._owner = client_id
      self._seq = seq
      self._seen = now
      self._axes = axes
      self._events.extend(sorted(buttons - self._buttons))
      self._buttons = buttons
      self._active = True

  def _release(self):
    self._stop_pending = self._active or self._stop_pending
    self._active = False
    self._owner = None
    self._seq = -1
    self._axes = (0.0, 0.0, 0.0)
    self._buttons.clear()
    stop_requested = "stop" in self._events
    self._events.clear()
    if stop_requested:
      self._events.append("stop")

  def poll(self):
    """Return latest velocity and button edges on the simulation thread."""
    with self._lock:
      if self._active and time.monotonic() - self._seen > self.TIMEOUT:
        self._release()
      events = list(self._events)
      self._events.clear()
      if self._active:
        forward, lateral, yaw = self._axes
        twist = (forward * (1.2 if forward >= 0 else 0.8), lateral * 0.5, yaw * 1.5)
        self._stop_pending = False
        return twist, events
      if self._stop_pending:
        self._stop_pending = False
        return (0.0, 0.0, 0.0), events
      return None, []

  def close(self):
    self.server._websock_server.unregister_handler(_messages.GuiUpdateMessage, self._receive)
