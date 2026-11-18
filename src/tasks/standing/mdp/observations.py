"""Standing task observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from src.tasks.standing.mdp.upper_body_motion import UpperBodyMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ref_upper_dof_pos(
  env: ManagerBasedRlEnv, command_name: str = "upper_body_motion"
) -> torch.Tensor:
  """FALCON-style upper-body reference joint positions (absolute rad)."""
  command = cast(UpperBodyMotionCommand, env.command_manager.get_term(command_name))
  return command.ref_joint_pos
