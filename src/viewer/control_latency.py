"""Two fixed, simulation-time delay lines: policy delivery and motor targets."""
from collections import deque
import math


class DelayLine:
  def __init__(self):
    self.pending = deque()
    self.initial = None

  def reset(self, initial):
    self.pending.clear()
    self.initial = initial.detach().clone()

  def push(self, value, steps):
    if steps == 0:
      return value
    self.pending.append(value.detach().clone())
    if len(self.pending) > steps:
      return self.pending.popleft()
    return self.initial.clone()


class ControlLatency:
  def __init__(self, controller):
    self.controller = controller
    self.env = controller.raw
    self.robot = self.env.scene['robot']
    self.inference_enabled = False
    self.motor_enabled = False
    self.inference_ms = 40.0
    self.motor_ms = 10.0
    self.scope = '全部模型'
    self.policy_line = DelayLine()
    self.motor_line = DelayLine()
    self.reset()
    # This hook runs once per physics step, after action scaling and before
    # scene.write_data_to_sim() converts targets into MuJoCo actuator controls.
    self._original_apply = self.env.action_manager.apply_action
    self.env.action_manager.apply_action = self._apply_motor

  @staticmethod
  def steps(milliseconds, dt):
    return max(0, math.ceil(milliseconds / (1000.0 * dt) - 1e-9))

  def reset(self):
    c = self.controller
    q = self.robot.data.joint_pos
    default = self.robot.data.default_joint_pos[:, c.joint_ids]
    # Seed with current pose, encoded in the new mode's action units.
    self.policy_line.reset((q[:, c.joint_ids] - default) / c._scale_buf)
    self.motor_line.reset(q)

  def configure(self, inference_enabled, inference_ms, motor_enabled, motor_ms, scope):
    if scope not in ('全部模型', '仅踢球'):
      raise ValueError('Invalid delay scope')
    for value in (inference_ms, motor_ms):
      if not math.isfinite(value) or not 0 <= value <= 200:
        raise ValueError('Delay must be in [0, 200] ms')
    self.inference_enabled = bool(inference_enabled)
    self.motor_enabled = bool(motor_enabled)
    self.inference_ms = float(inference_ms)
    self.motor_ms = float(motor_ms)
    self.scope = scope
    self.reset()

  @property
  def active(self):
    return (not self.controller.fall_guard.disabled and
            (self.scope == '全部模型' or self.controller.mode == 'kick'))

  def policy(self, action):
    steps = self.steps(self.inference_ms, self.env.step_dt) if self.active and self.inference_enabled else 0
    return self.policy_line.push(action, steps)

  def _apply_motor(self):
    self._original_apply()
    steps = self.steps(self.motor_ms, self.env.physics_dt) if self.active and self.motor_enabled else 0
    if steps:
      target = self.robot.data.joint_pos_target
      target[:] = self.motor_line.push(target, steps)

  @property
  def status(self):
    inference = self.steps(self.inference_ms, self.env.step_dt) * self.env.step_dt * 1000 if self.inference_enabled else 0
    motor = self.steps(self.motor_ms, self.env.physics_dt) * self.env.physics_dt * 1000 if self.motor_enabled else 0
    state = '当前生效' if self.active else '当前模式旁路 / 摔倒停控'
    return f'{self.scope} · {state}<br/>推理 {inference:g} ms + 电机 {motor:g} ms = {inference + motor:g} ms（仿真时间）'
