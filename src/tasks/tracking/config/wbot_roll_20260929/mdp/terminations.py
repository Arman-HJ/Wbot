from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def _persistence_buffer(
  env: ManagerBasedRlEnv, key: str
) -> torch.Tensor:
  """Per-env consecutive-fail counter, zeroed on episode reset."""
  buf = getattr(env, key, None)
  if buf is None or buf.shape[0] != env.num_envs:
    buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, key, buf)
  # Fresh episode → clear recovery streak.
  buf = torch.where(env.episode_length_buf == 0, torch.zeros_like(buf), buf)
  setattr(env, key, buf)
  return buf


def _with_recovery(
  env: ManagerBasedRlEnv,
  bad: torch.Tensor,
  *,
  key: str,
  recovery_steps: int,
) -> torch.Tensor:
  """Terminate only after ``recovery_steps`` consecutive bad frames.

  Good frames reset the streak, giving a recovery window after falls / tracking slips.
  """
  if recovery_steps <= 1:
    return bad
  buf = _persistence_buffer(env, key)
  buf = torch.where(bad, buf + 1, torch.zeros_like(buf))
  setattr(env, key, buf)
  return buf >= int(recovery_steps)


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  recovery_steps: int = 1,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  bad = (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )
  return _with_recovery(
    env, bad, key="_term_bad_anchor_pos_z", recovery_steps=recovery_steps
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
  recovery_steps: int = 1,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  bad = (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold
  return _with_recovery(
    env, bad, key="_term_bad_anchor_ori", recovery_steps=recovery_steps
  )


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  recovery_steps: int = 1,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  bad = torch.any(error > threshold, dim=-1)
  return _with_recovery(
    env, bad, key="_term_bad_ee_body_pos_z", recovery_steps=recovery_steps
  )
