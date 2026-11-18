"""Policy, critic, and discriminator observations for E1 locomotion AMP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Baked into ``_actor_frame`` (not ObservationTermCfg.scale). Deploy must apply these.
ACTOR_FRAME_ANG_VEL_SCALE = 0.2
ACTOR_FRAME_GRAVITY_SCALE = 1.0
ACTOR_FRAME_COMMAND_SCALE = 1.0
ACTOR_FRAME_JOINT_POS_SCALE = 1.0
ACTOR_FRAME_JOINT_VEL_SCALE = 0.05
ACTOR_FRAME_ACTION_SCALE = 1.0


def actor_frame_field_layout(num_joints: int) -> list[dict[str, Any]]:
  """Per-field slice / scale / train-noise for one packed actor frame."""
  q0 = 9
  dq0 = q0 + num_joints
  act0 = dq0 + num_joints
  act1 = act0 + num_joints
  return [
    {
      "name": "base_ang_vel",
      "slice": [0, 3],
      "dim": 3,
      "scale": ACTOR_FRAME_ANG_VEL_SCALE,
      "noise": {"type": "uniform", "n_min": -0.3 * ACTOR_FRAME_ANG_VEL_SCALE, "n_max": 0.3 * ACTOR_FRAME_ANG_VEL_SCALE},
      "note": "torso body-frame ang vel; value stored as scale * omega_b",
    },
    {
      "name": "projected_gravity",
      "slice": [3, 6],
      "dim": 3,
      "scale": ACTOR_FRAME_GRAVITY_SCALE,
      "noise": {"type": "uniform", "n_min": -0.05, "n_max": 0.05},
      "note": "torso body-frame gravity direction",
    },
    {
      "name": "command",
      "slice": [6, 9],
      "dim": 3,
      "scale": ACTOR_FRAME_COMMAND_SCALE,
      "noise": None,
      "note": "twist (vx, vy, wz)",
    },
    {
      "name": "joint_pos",
      "slice": [q0, dq0],
      "dim": num_joints,
      "scale": ACTOR_FRAME_JOINT_POS_SCALE,
      "noise": {"type": "uniform", "n_min": -0.02, "n_max": 0.02},
      "note": "q - default_joint_pos",
    },
    {
      "name": "joint_vel",
      "slice": [dq0, act0],
      "dim": num_joints,
      "scale": ACTOR_FRAME_JOINT_VEL_SCALE,
      "noise": {
        "type": "uniform",
        "n_min": -1.5 * ACTOR_FRAME_JOINT_VEL_SCALE,
        "n_max": 1.5 * ACTOR_FRAME_JOINT_VEL_SCALE,
      },
      "note": "value stored as scale * dq",
    },
    {
      "name": "actions",
      "slice": [act0, act1],
      "dim": num_joints,
      "scale": ACTOR_FRAME_ACTION_SCALE,
      "noise": None,
      "note": "previous network action (before action_scale)",
    },
  ]


def actor_frame_obs_scales(num_joints: int | None = None) -> dict[str, float]:
  del num_joints
  return {
    "base_ang_vel": ACTOR_FRAME_ANG_VEL_SCALE,
    "projected_gravity": ACTOR_FRAME_GRAVITY_SCALE,
    "command": ACTOR_FRAME_COMMAND_SCALE,
    "joint_pos": ACTOR_FRAME_JOINT_POS_SCALE,
    "joint_vel": ACTOR_FRAME_JOINT_VEL_SCALE,
    "actions": ACTOR_FRAME_ACTION_SCALE,
  }


def _ordered_joint_state(
  asset: Entity, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  default = asset.data.default_joint_pos
  assert default is not None
  ids = asset_cfg.joint_ids
  return (
    asset.data.joint_pos[:, ids],
    asset.data.joint_vel[:, ids],
    default[:, ids],
  )


def _torso_imu(
  asset: Entity, torso_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  quat_w = asset.data.body_link_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_w = asset.data.body_link_ang_vel_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
  gravity_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
  return ang_vel_b, gravity_b


def _actor_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  velocity_command_name: str,
  corrupt: bool,
) -> torch.Tensor:
  asset: Entity = env.scene[joint_cfg.name]
  ang_vel_b, gravity_b = _torso_imu(asset, torso_cfg)
  command = env.command_manager.get_command(velocity_command_name)
  assert command is not None
  joint_pos, joint_vel, default = _ordered_joint_state(asset, joint_cfg)
  action = env.action_manager.action

  frame = torch.cat(
    (
      ACTOR_FRAME_ANG_VEL_SCALE * ang_vel_b,
      ACTOR_FRAME_GRAVITY_SCALE * gravity_b,
      ACTOR_FRAME_COMMAND_SCALE * command,
      ACTOR_FRAME_JOINT_POS_SCALE * (joint_pos - default),
      ACTOR_FRAME_JOINT_VEL_SCALE * joint_vel,
      ACTOR_FRAME_ACTION_SCALE * action,
    ),
    dim=1,
  )
  num_joints = joint_pos.shape[1]
  assert frame.shape[1] == 9 + 3 * num_joints

  if corrupt:
    # Match the original Isaac Lab task's per-field uniform noise.  Keeping the
    # whole frame in one observation term preserves frame-major history order.
    noise = torch.zeros(frame.shape[1], device=env.device)
    for field in actor_frame_field_layout(num_joints):
      n = field.get("noise")
      if not n:
        continue
      start, end = field["slice"]
      noise[start:end] = float(n["n_max"])
    frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * noise
  return frame


def actor_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  velocity_command_name: str = "twist",
) -> torch.Tensor:
  corrupt = env.cfg.observations["actor"].enable_corruption
  return _actor_frame(
    env,
    joint_cfg,
    torso_cfg,
    velocity_command_name,
    corrupt=corrupt,
  )


def critic_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  feet_contact_sensor_name: str,
  velocity_command_name: str = "twist",
  landing_force_scale: float | None = None,
) -> torch.Tensor:
  """Actor frame + root lin vel (b) + foot contact [+ per-foot landing force].

  ``landing_force_scale`` appends ``|F| * first_contact * scale`` per foot (same
  quantity as ``soft_landing``); the symmetry helper must expect the extra 2 dims.
  """
  asset: Entity = env.scene[joint_cfg.name]
  actor = _actor_frame(
    env,
    joint_cfg,
    torso_cfg,
    velocity_command_name,
    corrupt=False,
  )
  sensor: ContactSensor = env.scene[feet_contact_sensor_name]
  assert sensor.data.found is not None
  found = sensor.data.found
  contact = (found > 0).float()
  if contact.ndim > 2:
    contact = contact.flatten(start_dim=2).any(dim=2).float()
  parts = [actor, asset.data.root_link_lin_vel_b, contact]
  if landing_force_scale is not None:
    assert sensor.data.force is not None
    force_mag = torch.norm(sensor.data.force, dim=-1)
    first_contact = sensor.compute_first_contact(dt=env.step_dt).float()
    parts.append(force_mag * first_contact * landing_force_scale)
  frame = torch.cat(parts, dim=1)
  extra = 2 if landing_force_scale is not None else 0
  assert frame.shape[1] == actor.shape[1] + 3 + 2 + extra
  return frame


def amp_state(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  key_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the discriminator state used by the expert NPZ loader."""
  asset: Entity = env.scene[joint_cfg.name]
  joint_pos, joint_vel, _ = _ordered_joint_state(asset, joint_cfg)
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  key_pos = asset.data.body_link_pos_w[:, key_body_cfg.body_ids, :]
  relative = key_pos - root_pos.unsqueeze(1)
  root_quat_repeated = root_quat.unsqueeze(1).expand(-1, relative.shape[1], -1)
  key_pos_b = quat_apply_inverse(root_quat_repeated, relative)
  key_quat_w = asset.data.body_link_quat_w[:, key_body_cfg.body_ids, :]
  _, key_quat_b = subtract_frame_transforms(
    root_pos.unsqueeze(1).expand_as(key_pos),
    root_quat_repeated,
    key_pos,
    key_quat_w,
  )
  key_ori_b = matrix_from_quat(key_quat_b)[..., :2]
  state = torch.cat(
    (
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_b,
      joint_pos,
      joint_vel,
      key_pos_b.flatten(start_dim=1),
      key_ori_b.flatten(start_dim=1),
    ),
    dim=1,
  )
  num_joints = joint_pos.shape[1]
  num_keys = key_pos.shape[1]
  assert state.shape[1] == 6 + 2 * num_joints + 9 * num_keys
  return state
