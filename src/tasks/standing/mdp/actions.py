"""Standing task actions (FALCON residual upper-body control)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names
from src.tasks.standing.mdp.upper_body_motion import UpperBodyMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualUpperBodyJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with FALCON-style residual upper-body reference.

  Lower-body joints: ``target = action * scale + default``.
  Upper-body joints: ``target = action * scale + ref_upper`` (from motion command).
  """

  upper_body_joint_names: tuple[str, ...] = ()
  upper_body_command_name: str = "upper_body_motion"

  def build(self, env: ManagerBasedRlEnv) -> ResidualUpperBodyJointPositionAction:
    return ResidualUpperBodyJointPositionAction(self, env)


class ResidualUpperBodyJointPositionAction(JointPositionAction):
  """Apply motion reference as the upper-body action offset (residual control)."""

  cfg: ResidualUpperBodyJointPositionActionCfg

  def __init__(self, cfg: ResidualUpperBodyJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._upper_action_indices: torch.Tensor | None = None
    if cfg.upper_body_joint_names:
      upper_ids, _ = resolve_matching_names(
        cfg.upper_body_joint_names, self._target_names, preserve_order=True
      )
      if upper_ids:
        self._upper_action_indices = torch.tensor(
          upper_ids, dtype=torch.long, device=self.device
        )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias

    if self._upper_action_indices is not None and len(self._upper_action_indices) > 0:
      command = cast(
        UpperBodyMotionCommand,
        self._env.command_manager.get_term(self.cfg.upper_body_command_name),
      )
      for i, action_idx in enumerate(self._upper_action_indices):
        scale_i = (
          self._scale[:, action_idx]
          if isinstance(self._scale, torch.Tensor)
          else self._scale
        )
        target[:, action_idx] = (
          self._raw_actions[:, action_idx] * scale_i
          + command.ref_joint_pos[:, i]
          - encoder_bias[:, action_idx]
        )

    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
