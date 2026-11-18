"""Build Adam SP 12-DoF legs-only MjSpec by baking waist/arms rigid."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from src.assets.robots.adam_sp.adam_sp_constants import ADAM_SP_XML, HOME_KEYFRAME
from src.assets.robots.spec.frozen_joint_spec import build_spec_with_frozen_joints

# Keep 12 leg joints; bake waist + arms (+ wrists) at HOME_KEYFRAME pose.
ADAM_SP_UPPER_BODY_REMOVE_RE = re.compile(
  r"^(waist_.*_joint|"
  r"(left|right)_shoulder_.*_joint|"
  r"(left|right)_elbow_joint|"
  r"(left|right)_wrist_.*_joint)$"
)


def _expand_keyframe_joint_pos(
  source_xml: Path,
  joint_pos: dict[str, float],
) -> dict[str, float]:
  """Expand regex keyframe entries to exact joint names for baking."""
  spec = mujoco.MjSpec.from_file(str(source_xml))
  names = [j.name for j in spec.joints if j.name and j.name != "root"]
  expanded: dict[str, float] = {}
  for pattern, value in joint_pos.items():
    for name in names:
      if re.fullmatch(pattern, name):
        expanded[name] = float(value)
  return expanded


def build_adam_sp_legs_12dof_spec() -> mujoco.MjSpec:
  """Waist/arms rigid at HOME_KEYFRAME; keep 12 leg DoFs."""
  assert HOME_KEYFRAME.joint_pos is not None
  bake_joint_pos = _expand_keyframe_joint_pos(
    ADAM_SP_XML, dict(HOME_KEYFRAME.joint_pos)
  )
  bake_keyframe = EntityCfg.InitialStateCfg(
    pos=HOME_KEYFRAME.pos,
    joint_pos=bake_joint_pos,
    joint_vel={".*": 0.0},
  )
  return build_spec_with_frozen_joints(
    ADAM_SP_UPPER_BODY_REMOVE_RE,
    source_xml=ADAM_SP_XML,
    locomotion_keyframe=bake_keyframe,
    spec_log_label="Adam SP legs 12DoF",
  )
