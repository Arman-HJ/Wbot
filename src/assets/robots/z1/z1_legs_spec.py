"""Build legs+waist Z1 MjSpec by baking upper-body pose and removing joints."""

from __future__ import annotations

import mujoco

from src.assets.robots.spec.frozen_joint_spec import (
  ARMS_HANDS_NECK_REMOVE_RE,
  build_spec_with_frozen_joints,
)
from src.assets.robots.z1.z1_constants import (
  Z1_LEGS_ONLY_XML,
  Z1_LOCOMOTION_KEYFRAME,
)


def build_z1_legs_waist_spec() -> mujoco.MjSpec:
  """Arms/hands/neck rigid at Z1_LOCOMOTION_KEYFRAME; keep legs + waist DoFs."""
  return build_spec_with_frozen_joints(
    ARMS_HANDS_NECK_REMOVE_RE,
    source_xml=Z1_LEGS_ONLY_XML,
    locomotion_keyframe=Z1_LOCOMOTION_KEYFRAME,
    spec_log_label="Z1 legs",
  )
