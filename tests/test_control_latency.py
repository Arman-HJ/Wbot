from types import SimpleNamespace
import torch
from src.viewer.control_latency import ControlLatency, DelayLine


def make_controller():
  data = SimpleNamespace(joint_pos=torch.tensor([[.1, .2]]), default_joint_pos=torch.zeros(1, 2), joint_pos_target=torch.zeros(1, 2))
  def apply():
    data.joint_pos_target[:] = torch.tensor([[.8, .9]])
  raw = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}, step_dt=.02, physics_dt=.005, action_manager=SimpleNamespace(apply_action=apply))
  c = SimpleNamespace(raw=raw, joint_ids=[0, 1], _scale_buf=torch.tensor([[.25, .25]]), mode='kick', fall_guard=SimpleNamespace(disabled=False))
  c.latency = ControlLatency(c)
  return c


def test_delay_line_timing_and_snapshot():
  line = DelayLine()
  line.reset(torch.tensor([0.]))
  value = torch.tensor([1.])
  assert line.push(value, 2).item() == 0
  value.fill_(2)
  assert line.push(value, 2).item() == 0
  value.fill_(3)
  assert line.push(value, 2).item() == 1
  line.reset(torch.tensor([7.]))
  assert line.push(value, 2).item() == 7
  assert line.push(value, 0) is value


def test_motor_physics_steps_and_policy_control_steps():
  c = make_controller(); delay = c.latency
  delay.configure(True, 40, True, 10, '全部模型')
  a = torch.ones(1, 2)
  for _ in range(2):
    torch.testing.assert_close(delay.policy(a), torch.tensor([[.4, .8]]))
  torch.testing.assert_close(delay.policy(a), a)
  for _ in range(2):
    c.raw.action_manager.apply_action()
    torch.testing.assert_close(c.raw.scene['robot'].data.joint_pos_target, torch.tensor([[.1, .2]]))
  c.raw.action_manager.apply_action()
  torch.testing.assert_close(c.raw.scene['robot'].data.joint_pos_target, torch.tensor([[.8, .9]]))


def test_scope_guard_reset_and_zero_bypass():
  c = make_controller(); delay = c.latency; a = torch.ones(1, 2)
  delay.configure(True, 40, True, 10, '仅踢球')
  c.mode = 'walk'
  assert delay.policy(a) is a
  c.raw.action_manager.apply_action()
  torch.testing.assert_close(c.raw.scene['robot'].data.joint_pos_target, torch.tensor([[.8, .9]]))
  c.mode = 'kick'; c.fall_guard.disabled = True
  assert delay.policy(a) is a
  c.fall_guard.disabled = False
  delay.configure(True, 0, True, 0, '全部模型')
  assert delay.policy(a) is a
  assert ControlLatency.steps(21, .02) == 2
