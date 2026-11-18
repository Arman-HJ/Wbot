"""Standing reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from src.tasks.standing.mdp.leg_pitch_ik import LegPitchIkParams, leg_pitch_ik_from_base_height

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class ResetStandingHeightFromCommand(ManagerTermBase):
  """Align leg/waist spawn pose with the resampled ``base_height`` command.

  Uses the same planar 2-link leg pitch IK as the height-aware rewards.
  """

  cfg: EventTermCfg

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.cfg = cfg
    params = cfg.params
    self._height_command_name = params.get("height_command_name", "base_height")

    self._asset_cfg: SceneEntityCfg = params.get(
      "asset_cfg", SceneEntityCfg("robot")
    )
    self._asset_cfg.resolve(env.scene)
    asset: Entity = env.scene[self._asset_cfg.name]

    joint_cfg: SceneEntityCfg = params.get(
      "joint_asset_cfg",
      SceneEntityCfg(
        "robot",
        joint_names=(
          r".*_hip_pitch_joint",
          r".*_knee_joint",
          r".*_ankle_pitch_joint",
        ),
      ),
    )
    joint_cfg.resolve(env.scene)
    assert joint_cfg.joint_ids is not None
    self._joint_ids = joint_cfg.joint_ids
    _, joint_names = asset.find_joints(joint_cfg.joint_names, preserve_order=True)
    self._joint_names = list(joint_names)

    defaults = LegPitchIkParams()
    self._ik_params = LegPitchIkParams(
      thigh_length=float(params.get("thigh_length", defaults.thigh_length)),
      shank_length=float(params.get("shank_length", defaults.shank_length)),
      torso_hip_offset=float(
        params.get("torso_hip_offset", defaults.torso_hip_offset)
      ),
    )

    pitch_patterns = (
      r".*_hip_pitch_joint",
      r".*_knee_joint",
      r".*_ankle_pitch_joint",
    )
    self._pitch_local_indices: list[list[int]] = []
    for pattern in pitch_patterns:
      _, names = asset.find_joints(pattern, preserve_order=True)
      local = [self._joint_names.index(n) for n in names if n in self._joint_names]
      self._pitch_local_indices.append(local)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    del env, env_ids, kwargs

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = resolve_env_ids(self._env, env_ids)
    if len(env_ids) == 0:
      return

    height_cmd = self._env.command_manager.get_command(self._height_command_name)
    if height_cmd is None:
      return

    asset: Entity = self._env.scene[self._asset_cfg.name]
    hip_ref, knee_ref, ankle_ref = leg_pitch_ik_from_base_height(
      height_cmd[env_ids, 0], self._ik_params
    )
    refs = (hip_ref, knee_ref, ankle_ref)

    default_joint = asset.data.default_joint_pos[env_ids][:, self._joint_ids].clone()
    for local_indices, values in zip(self._pitch_local_indices, refs, strict=True):
      for local_idx in local_indices:
        default_joint[:, local_idx] = values

    joint_vel = torch.zeros_like(default_joint)
    asset.write_joint_state_to_sim(
      default_joint, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids
    )
