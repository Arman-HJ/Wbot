"""Mutually exclusive standing, forward-only and general WBOT commands."""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class WbotVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    category = torch.rand(len(env_ids), device=self.device)
    self.is_standing_env[env_ids] = category < self.cfg.rel_standing_envs
    self.is_forward_env[env_ids] = (
      (category >= self.cfg.rel_standing_envs)
      & (category < self.cfg.rel_standing_envs + self.cfg.rel_forward_envs)
    )
    forward = env_ids[self.is_forward_env[env_ids]]
    self.is_heading_env[forward] = False
    self.is_world_env[forward] = False
    if len(forward):
      maximum = max(0.1, self.cfg.ranges.lin_vel_x[1])
      minimum = min(0.3, maximum)
      self.vel_command_b[forward, 0] = torch.empty(len(forward),device=self.device).uniform_(minimum, maximum)
      self.vel_command_b[forward, 1:] = 0.0

  def _update_command(self):
    super()._update_command()
    self.vel_command_b[self.is_forward_env, 1:] = 0.0


@dataclass(kw_only=True)
class WbotVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return WbotVelocityCommand(self, env)
