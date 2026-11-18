"""Isolated, contact-validated supine-to-prone tracking task."""
from dataclasses import dataclass, fields
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import torch
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.spec_config import CollisionCfg
from .robot_constants import get_assets
from src.tasks.tracking.config.wbot_roll_20260929.mdp.commands import MotionCommand
from src.tasks.tracking.config.wbot_roll_20260929.mdp.terminations import _with_recovery
from .env_cfgs import wbot_20dof_getup_tracking_env_cfg
from .rl_cfg import wbot_20dof_getup_tracking_ppo_runner_cfg

MOTION_DIR = Path(__file__).resolve().parents[4] / 'assets/motions/wbot/roll_to_prone_20260929'


def roll_robot_spec():
  tree = ET.parse(MOTION_DIR / 'simulation_model.xml')
  root = tree.getroot()
  # The scene supplies the ground and articulation supplies the servos.
  for node in list(root):
    if node.tag in ('actuator', 'keyframe', 'option'):
      root.remove(node)
  world = root.find('worldbody')
  for node in list(world):
    if node.tag == 'geom':
      world.remove(node)
  for parent in root.iter():
    for node in list(parent):
      if node.tag in ('light', 'camera'):
        parent.remove(node)
  spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
  spec.assets = get_assets(spec.meshdir)
  return spec


class RollMotionCommand(MotionCommand):
  def _update_command(self):
    # Hold the terminal reference; never teleport back to supine at clip end.
    self.time_steps.clamp_(max=self.motion.time_step_total - 2)
    super()._update_command()


@dataclass(kw_only=True)
class RollMotionCommandCfg(MotionCommandCfg):
  start_time_step: int = 0
  start_time_step_prob: float = 0.5

  def build(self, env):
    return RollMotionCommand(self, env)


def roll_tilt_failure(env, asset_cfg, command_name, threshold, recovery_steps=5):
  command = env.command_manager.get_term(command_name)
  gravity = env.scene[asset_cfg.name].data.gravity_vec_w
  target = quat_apply_inverse(command.anchor_quat_w, gravity)
  actual = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
  # All three components distinguish prone from supine, unlike gravity Z alone.
  bad = torch.linalg.vector_norm(target - actual, dim=-1) > threshold
  return _with_recovery(env, bad, key='_roll_tilt_failure', recovery_steps=recovery_steps)


def roll_env_cfg(play=False):
  cfg = wbot_20dof_getup_tracking_env_cfg(play=play)
  cfg.episode_length_s = 3.0 if not play else 1e9
  cfg.sim.mujoco.timestep = 0.004
  cfg.sim.mujoco.iterations = 40
  cfg.decimation = 5
  robot = cfg.scene.entities['robot']
  robot.spec_fn = roll_robot_spec
  robot.collisions = ()  # Keep ground-only mesh proxies and original self collisions.
  robot.articulation.soft_joint_pos_limit_factor = 1.0
  cfg.scene.terrain.collisions = (CollisionCfg(
    geom_names_expr=('terrain',), contype=1, conaffinity=3, condim=3,
    friction=(1.0, 0.005, 0.0001), solref=(0.008, 1.0),
  ),)
  old = cfg.commands['motion']
  motion = RollMotionCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  motion.motion_file = str(MOTION_DIR / 'tracking_motion.npz')
  motion.pose_range = {} if play else {'x': (-0.01, 0.01), 'y': (-0.01, 0.01), 'yaw': (-0.05, 0.05)}
  motion.velocity_range = {} if play else {'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}
  motion.joint_position_range = (0.0, 0.0) if play else (-0.03, 0.03)
  cfg.commands['motion'] = motion
  cfg.events.pop('push_robot', None)
  cfg.events['base_com'].params['ranges'] = {i: (-0.005, 0.005) for i in range(3)}
  cfg.events['foot_friction'].params['asset_cfg'].geom_names = r'^(left|right)_(foot[1-7]_collision|ankle_roll_ground_mesh_0)$'
  cfg.events['foot_friction'].params['ranges'] = (0.55, 0.65)
  cfg.terminations['anchor_ori'].func = roll_tilt_failure
  cfg.terminations['anchor_ori'].params.update(threshold=1.2, recovery_steps=5)
  cfg.viewer.distance = 1.8
  cfg.viewer.elevation = -15.0
  return cfg


def roll_rl_cfg():
  cfg = wbot_20dof_getup_tracking_ppo_runner_cfg()
  cfg.experiment_name = 'wbot_20dof_roll_to_prone_tracking'
  cfg.wandb_project = cfg.experiment_name
  cfg.max_iterations = 5001
  cfg.save_interval = 250
  return cfg
