"""Getup task rewards / terminations / events (from smp getup, no SMP gate).

Mirrors ``smp.rl.tasks.getup`` task shaping used by ``getup_env_cfg.py``, but the
SMP product gate is dropped: AMP supplies the style prior instead.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
  "track_head_height",
  "upward_velocity",
  "getup_task_reward",
  "pose",
  "stood_up",
  "reset_stand_counter",
  "plant_root_above_ground",
]


def _head_index(env: ManagerBasedRlEnv) -> int:
  robot = env.scene["robot"]
  return int(robot.find_sites(["head"], preserve_order=True)[0][0])


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 0.85,
  scale: float = 1.0,
) -> torch.Tensor:
  """``exp(-scale·max(target_height − head_z, 0)²)`` — same as smp getup."""
  robot = env.scene["robot"]
  z = robot.data.site_pos_w[:, _head_index(env), 2]
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.65,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward head velocity while still low; saturate once tall enough."""
  robot = env.scene["robot"]
  head_idx = _head_index(env)
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(head_z < head_height_threshold, shaped, torch.ones_like(shaped))


def getup_task_reward(
  env: ManagerBasedRlEnv,
  upward_weight: float = 0.7,
  height_weight: float = 0.3,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.65,
  upward_scale: float = 100.0,
  target_height: float = 0.85,
  height_scale: float = 1.0,
) -> torch.Tensor:
  """``0.7·upward_velocity + 0.3·track_head_height`` from getup_env_cfg (no SMP)."""
  return upward_weight * upward_velocity(
    env,
    target_velocity=target_velocity,
    head_height_threshold=head_height_threshold,
    scale=upward_scale,
  ) + height_weight * track_head_height(
    env, target_height=target_height, scale=height_scale
  )


def pose(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  std: dict[str, float],
  min_head_height: float = 0.80,
) -> torch.Tensor:
  """Default-pose reward once the head is high enough.

  Same form as velocity ``variable_posture`` standing branch:
  ``exp(−mean((q−q₀)² / σ²))`` with per-joint σ from ``std``, but no
  walk/run switching. Gated by head height so get-up is not pulled home early.
  Use a **positive** weight.
  """
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  robot = env.scene["robot"]
  default = robot.data.default_joint_pos
  assert default is not None

  _, joint_names = robot.find_joints(asset_cfg.joint_names, preserve_order=True)
  _, _, std_vals = resolve_matching_names_values(
    data=std,
    list_of_strings=list(joint_names),
  )
  std_t = torch.tensor(std_vals, device=env.device, dtype=torch.float32)

  current = robot.data.joint_pos[:, asset_cfg.joint_ids]
  target = default[:, asset_cfg.joint_ids]
  error_sq = torch.square(current - target)
  reward = torch.exp(-torch.mean(error_sq / (std_t * std_t), dim=-1))

  head_z = robot.data.site_pos_w[:, _head_index(env), 2]
  return torch.where(head_z >= min_head_height, reward, torch.zeros_like(reward))


# Back-compat alias.
stand_still = pose


def stood_up(
  env: ManagerBasedRlEnv,
  head_height: float = 0.88,
  max_speed: float = 0.5,
  hold_steps: int = 25,
) -> torch.Tensor:
  """Success truncation once stably standing (``time_out=True`` in cfg)."""
  robot = env.scene["robot"]
  z = robot.data.site_pos_w[:, _head_index(env), 2]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  is_standing = (z >= head_height) & (speed < max_speed)
  cnt = getattr(env, "_getup_stand_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
  env._getup_stand_count = cnt  # type: ignore[attr-defined]
  return cnt >= hold_steps


@torch.no_grad()
def reset_stand_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Zero the ``stood_up`` hold counter on reset."""
  if not hasattr(env, "_getup_stand_count"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._getup_stand_count[env_ids] = 0  # type: ignore[attr-defined]


@torch.no_grad()
def plant_root_above_ground(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  clearance: float = 0.01,
) -> None:
  """After a fallen-pose reset, lift the root so the lowest body clears z=0.

  Get-up randomizes large roll/pitch; a fixed root-z offset then buries limbs
  underground. This only shifts root height — orientation is unchanged.

  Must forward kinematics first: prior reset events write qpos, but derived
  ``body_link_pos_w`` is stale until ``sim.forward()``.
  """
  from mjlab.envs.mdp.events import resolve_env_ids

  env_ids = resolve_env_ids(env, env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  robot = env.scene["robot"]
  body_z = robot.data.body_link_pos_w[env_ids, :, 2]
  lowest = body_z.min(dim=-1).values
  lift = clearance - lowest
  pose = robot.data.root_link_pose_w[env_ids].clone()
  pose[:, 2] += lift
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
