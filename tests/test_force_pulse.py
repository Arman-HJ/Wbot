import math
from types import SimpleNamespace
import unittest
import torch
from src.viewer.force_pulse import ForcePulse


class ForcePulseTest(unittest.TestCase):
  def setUp(self):
    self.calls = []
    robot = SimpleNamespace(
      body_names=["pelvis", "torso_link"],
      data=SimpleNamespace(root_link_quat_w=torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])),
      write_external_wrench_to_sim=lambda f, t, **kw: self.calls.append((f.clone(), t.clone(), kw)),
    )
    self.push = ForcePulse(SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=1, step_dt=.02))

  def test_world_direction_and_step_duration(self):
    self.push.start("向前", 50., .06)
    for i in range(3):
      self.push.before_step()
      torch.testing.assert_close(self.calls[-1][0], torch.tensor([[[0., 50., 0.]]]), atol=1e-5, rtol=1e-5)
      self.assertEqual(self.push.remaining, 3-i)
      self.push.after_step()
    self.assertEqual(self.push.remaining, 0)
    self.assertFalse(self.calls[-1][0].any())
    self.assertEqual(self.calls[-1][2]["body_ids"], [1])

  def test_replace_and_cancel_do_not_leave_force(self):
    self.push.start("向前", 50., 1.)
    self.push.before_step()
    self.push.start("向右", 10., .1)
    self.assertFalse(self.calls[-1][0].any())
    self.push.before_step()
    torch.testing.assert_close(self.calls[-1][0], torch.tensor([[[10., 0., 0.]]]), atol=1e-5, rtol=1e-5)
    self.push.cancel()
    self.assertFalse(self.calls[-1][0].any())
    count = len(self.calls)
    self.push.before_step()
    self.assertEqual(len(self.calls), count)

  def test_reject_invalid_input(self):
    for force, duration in ((float("nan"), .2), (301., .2), (60., 0.)):
      with self.assertRaises(ValueError):
        self.push.start("向前", force, duration)
    self.assertEqual(self.calls, [])


if __name__ == "__main__":
  unittest.main()
