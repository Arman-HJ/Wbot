"""Left-right symmetry for the WBOT AMP tasks (20-DoF and leg-only 12-DoF).

Same reflection as the E1 helper, without the waist joint. Frame widths follow
the actuated joints and AMP key bodies of each variant.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.amp.wbot_constants import (
  WBOT_12DOF_JOINT_NAMES,
  WBOT_12DOF_KEY_BODY_NAMES,
  WBOT_JOINT_NAMES,
  WBOT_KEY_BODY_NAMES,
)

_POLAR_SIGN = (1.0, -1.0, 1.0)
_AXIAL_SIGN = (-1.0, 1.0, -1.0)
_COMMAND_SIGN = (1.0, -1.0, -1.0)


def _mirror_name(name: str) -> str:
  if name.startswith("left_"):
    return name.replace("left_", "right_", 1)
  if name.startswith("right_"):
    return name.replace("right_", "left_", 1)
  return name


class _WbotMirror:
  def __init__(
    self,
    joint_names: tuple[str, ...],
    key_body_names: tuple[str, ...],
    critic_landing_force: bool = False,
  ) -> None:
    self.num_joints = len(joint_names)
    self.num_keys = len(key_body_names)
    self.actor_frame_dim = 9 + 3 * self.num_joints
    # + root lin vel (3) + foot contact (2) [+ per-foot landing force (2)]
    self.critic_landing_force = critic_landing_force
    self.critic_frame_dim = self.actor_frame_dim + 5 + (2 if critic_landing_force else 0)
    self.amp_obs_dim = 6 + 2 * self.num_joints + 9 * self.num_keys
    self.joint_mirror = tuple(
      joint_names.index(_mirror_name(name)) for name in joint_names
    )
    self.joint_sign = tuple(
      -1.0 if any(axis in name for axis in ("roll", "yaw")) else 1.0
      for name in joint_names
    )
    self.key_mirror = tuple(
      key_body_names.index(_mirror_name(name)) for name in key_body_names
    )

  def joints(self, value: torch.Tensor) -> torch.Tensor:
    index = torch.tensor(self.joint_mirror, device=value.device)
    sign = value.new_tensor(self.joint_sign)
    return value[..., index] * sign

  def actor(self, value: torch.Tensor) -> torch.Tensor:
    dim = self.actor_frame_dim
    if value.shape[-1] % dim:
      raise ValueError(f"Actor width {value.shape[-1]} is not frame-aligned.")
    q0, dq0 = 9, 9 + self.num_joints
    act0 = dq0 + self.num_joints
    frames = value.reshape(*value.shape[:-1], -1, dim)
    out = frames.clone()
    out[..., 0:3] *= frames.new_tensor(_AXIAL_SIGN)
    out[..., 3:6] *= frames.new_tensor(_POLAR_SIGN)
    out[..., 6:9] *= frames.new_tensor(_COMMAND_SIGN)
    out[..., q0:dq0] = self.joints(frames[..., q0:dq0])
    out[..., dq0:act0] = self.joints(frames[..., dq0:act0])
    out[..., act0:] = self.joints(frames[..., act0:])
    return out.reshape_as(value)

  def critic(self, value: torch.Tensor) -> torch.Tensor:
    dim = self.critic_frame_dim
    if value.shape[-1] % dim:
      raise ValueError(f"Critic width {value.shape[-1]} is not frame-aligned.")
    frames = value.reshape(*value.shape[:-1], -1, dim)
    out = frames.clone()
    lin = self.actor_frame_dim
    out[..., :lin] = self.actor(frames[..., :lin])
    out[..., lin : lin + 3] *= frames.new_tensor(_POLAR_SIGN)
    out[..., lin + 3 : lin + 5] = frames[..., (lin + 4, lin + 3)]
    if self.critic_landing_force:
      out[..., lin + 5 : lin + 7] = frames[..., (lin + 6, lin + 5)]
    return out.reshape_as(value)

  def amp(self, value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != self.amp_obs_dim:
      raise ValueError(
        f"AMP width must be {self.amp_obs_dim}, got {value.shape[-1]}."
      )
    q0, dq0 = 6, 6 + self.num_joints
    key0 = dq0 + self.num_joints
    key_pos_end = key0 + 3 * self.num_keys
    key_ori_end = key_pos_end + 6 * self.num_keys
    batch = value.shape[:-1]
    out = value.clone()
    out[..., 0:3] *= value.new_tensor(_POLAR_SIGN)
    out[..., 3:6] *= value.new_tensor(_AXIAL_SIGN)
    out[..., q0:dq0] = self.joints(value[..., q0:dq0])
    out[..., dq0:key0] = self.joints(value[..., dq0:key0])
    key = value[..., key0:key_pos_end].reshape(*batch, self.num_keys, 3)
    key = key[..., self.key_mirror, :] * value.new_tensor(_POLAR_SIGN)
    out[..., key0:key_pos_end] = key.reshape(*batch, 3 * self.num_keys)
    key_ori = value[..., key_pos_end:key_ori_end].reshape(
      *batch, self.num_keys, 3, 2
    )
    key_ori = key_ori[..., self.key_mirror, :, :]
    ori_sign = value.new_tensor(((1.0, -1.0), (-1.0, 1.0), (1.0, -1.0)))
    out[..., key_pos_end:key_ori_end] = (key_ori * ori_sign).reshape(
      *batch, 6 * self.num_keys
    )
    return out

  def augment(
    self,
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
  ) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append mirrored samples using the RSL-RL 5.4 TensorDict interface."""
    del env
    obs_aug = None
    if obs is not None:
      mirrored = {}
      for key, value in obs.items():
        if key == "actor":
          mirrored[key] = self.actor(value)
        elif key == "critic":
          mirrored[key] = self.critic(value)
        elif key == "amp":
          mirrored[key] = self.amp(value)
        else:
          mirrored[key] = value.clone()
      obs_aug = TensorDict(
        {key: torch.cat((obs[key], mirrored[key]), dim=0) for key in obs.keys()},
        batch_size=[2 * obs.batch_size[0]],
        device=obs.device,
      )

    actions_aug = None
    if actions is not None:
      actions_aug = torch.cat((actions, self.joints(actions)), dim=0)
    return obs_aug, actions_aug


_MIRROR_20DOF = _WbotMirror(WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES)
_MIRROR_20DOF_LANDING = _WbotMirror(
  WBOT_JOINT_NAMES, WBOT_KEY_BODY_NAMES, critic_landing_force=True
)
_MIRROR_12DOF = _WbotMirror(WBOT_12DOF_JOINT_NAMES, WBOT_12DOF_KEY_BODY_NAMES)

data_augmentation_func = _MIRROR_20DOF.augment
data_augmentation_func_landing = _MIRROR_20DOF_LANDING.augment
data_augmentation_func_12dof = _MIRROR_12DOF.augment
