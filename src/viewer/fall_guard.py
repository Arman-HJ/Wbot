"""Latched fall detection and passive actuator control for the shared scene."""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


class FallGuard:
  def __init__(self, env):
    self.env = env
    self.robot = env.scene["robot"]
    self.torso_id = list(self.robot.body_names).index("torso_link")
    env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))
    self.ctrl_ids = [i for actuator in self.robot.actuators for i in actuator.global_ctrl_ids]
    self.defaults = {
      name: getattr(env.sim.model, name)[:, self.ctrl_ids].clone()
      for name in ("actuator_gainprm", "actuator_biasprm")
    }
    self.enabled = True
    self.disabled = False
    self.elapsed = torch.zeros(env.num_envs, device=env.device)
    self.reason = ""

  def restore(self):
    if self.disabled:
      for name, values in self.defaults.items():
        getattr(self.env.sim.model, name)[:, self.ctrl_ids] = values
    self.disabled = False
    self.elapsed.zero_()
    self.reason = ""

  def check(self, mode):
    if self.disabled:
      return True
    if not self.enabled or mode not in ("walk", "kick"):
      self.elapsed.zero_()
      return False
    gravity = quat_apply_inverse(
      self.robot.data.body_link_quat_w[:, self.torso_id], self.robot.data.gravity_vec_w,
    )
    cosine = -gravity[:, 2] / gravity.norm(dim=-1).clamp_min(1e-6)
    height = self.robot.data.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
    fallen = (cosine < math.cos(math.radians(60))) | (height < 0.30)
    self.elapsed[:] = torch.where(fallen, self.elapsed + self.env.step_dt, 0.)
    indices = (self.elapsed >= 0.1 - 1e-6).nonzero().flatten()
    if not len(indices):
      return False
    index = int(indices[0])
    angle = math.degrees(math.acos(float(cosine[index].clamp(-1, 1))))
    self.reason = f"已取消模型控制：环境 {index} 倾角 {angle:.0f}°，骨盆高度 {height[index]:.2f} m；选择模式或重置恢复"
    for name in self.defaults:
      getattr(self.env.sim.model, name)[:, self.ctrl_ids] = 0
    self.disabled = True
    return True
