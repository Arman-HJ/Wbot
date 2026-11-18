"""Control lease, stop, validation and edge behavior without GPU simulation."""
import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from viser import _messages
from src.viewer.gamepad import BrowserGamepad


class BrowserGamepadTest(unittest.TestCase):
  def setUp(self):
    self.control = BrowserGamepad(MagicMock())

  def send(self, seq=1, axes=(1, -1, 1), buttons=(), enabled=True, client=1):
    message = _messages.GuiUpdateMessage(self.control.token, {
      "value": json.dumps(dict(seq=seq, axes=axes, buttons=buttons, enabled=enabled))
    })
    asyncio.run(self.control._receive(client, message))

  def test_axis_limits_and_invalid_sample(self):
    self.send(axes=(2, -3, 4))
    self.assertEqual(self.control.poll()[0], (1.2, -0.5, 1.5))
    self.send(seq=2, axes=(float("nan"), 0, 0))
    self.assertEqual(self.control.poll()[0], (1.2, -0.5, 1.5))

  def test_owner_and_sequence(self):
    self.send(seq=2)
    self.send(seq=3, axes=(0, 0, 0), client=2)
    self.send(seq=1, axes=(0, 0, 0))
    self.assertEqual(self.control.poll()[0], (1.2, -0.5, 1.5))

  def test_button_edges_and_stop_before_release(self):
    self.send(buttons=("kick",))
    self.assertEqual(self.control.poll()[1], ["kick"])
    self.send(seq=2, buttons=("kick",))
    self.assertEqual(self.control.poll()[1], [])
    self.send(seq=3, buttons=("stop",))
    self.send(seq=4, enabled=False)
    self.assertEqual(self.control.poll(), ((0., 0., 0.), ["stop"]))
    self.assertEqual(self.control.poll(), (None, []))

  def test_getup_is_a_single_button_edge(self):
    self.send(buttons=("getup",))
    self.assertEqual(self.control.poll()[1], ["getup"])
    self.send(seq=2, buttons=("getup",))
    self.assertEqual(self.control.poll()[1], [])

  def test_roll_button_is_a_single_edge(self):
    self.send(buttons=("roll",))
    self.assertEqual(self.control.poll()[1], ["roll"])
    self.send(seq=2, buttons=("roll",))
    self.assertEqual(self.control.poll()[1], [])

  def test_watchdog_then_new_owner(self):
    with patch("src.viewer.gamepad.time.monotonic", return_value=10):
      self.send()
    with patch("src.viewer.gamepad.time.monotonic", return_value=11):
      self.assertEqual(self.control.poll(), ((0., 0., 0.), []))
      self.send(client=2, axes=(-1, 0, 0))
      self.assertEqual(self.control.poll()[0], (-0.8, 0., 0.))


if __name__ == "__main__":
  unittest.main()
