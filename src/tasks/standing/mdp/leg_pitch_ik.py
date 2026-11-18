"""Planar 2-link leg pitch IK from commanded base (torso) height.

Standing height tracking changes mainly through sagittal hip/knee/ankle pitch
while feet stay on the ground. We approximate each leg as a planar serial chain
(thigh + shank) and solve inverse kinematics for the pitch joints.

Geometry (foot placed vertically below the hip pitch axis):
  d = H_base - z_offset          # hip-to-foot vertical reach
  cos(q_knee) = (L1^2 + L2^2 - d^2) / (2 L1 L2)
  q_knee = pi - acos(cos_q_knee) # knee flexion (MuJoCo positive = flex)
  q_hip  = -atan2(L2 sin q_knee, L1 + L2 cos q_knee)
  q_ankle = -(q_hip + q_knee)    # flat-foot constraint
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LegPitchIkParams:
  """Calibrated planar leg parameters (Adam SP defaults)."""

  thigh_length: float = 0.4618
  shank_length: float = 0.3700
  """Vertical offset from torso height command to hip-foot reach."""
  torso_hip_offset: float = 0.2103
  min_hip_foot_reach: float | None = None
  max_hip_foot_reach: float | None = None


def leg_pitch_ik_from_base_height(
  base_height: torch.Tensor,
  params: LegPitchIkParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Map commanded torso clearance height to leg pitch angles.

  Args:
    base_height: Target torso height above ground ``(N,)`` or ``(N, 1)``.
    params: Link lengths and torso/hip offset.

  Returns:
    ``(hip_pitch, knee, ankle_pitch)`` each of shape ``(N,)``.
  """
  if params is None:
    params = LegPitchIkParams()

  height = base_height.reshape(-1).to(dtype=torch.float32)
  l1 = float(params.thigh_length)
  l2 = float(params.shank_length)
  reach = height - float(params.torso_hip_offset)

  min_reach = float(params.min_hip_foot_reach or (abs(l1 - l2) + 0.02))
  max_reach = float(params.max_hip_foot_reach or (l1 + l2 - 0.02))
  reach = torch.clamp(reach, min=min_reach, max=max_reach)

  cos_knee = (l1 * l1 + l2 * l2 - reach * reach) / (2.0 * l1 * l2)
  cos_knee = torch.clamp(cos_knee, -1.0, 1.0)
  knee = torch.pi - torch.acos(cos_knee)

  beta = torch.atan2(l2 * torch.sin(knee), l1 + l2 * torch.cos(knee))
  hip = -beta
  ankle = -(hip + knee)
  return hip, knee, ankle
