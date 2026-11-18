"""Frozen 2026-09-28 Roban WBOT 20DoF pin (widened hip-pitch limits).

Shares actuators / home / collisions with ``robots.wbot``; only the MJCF path
differs so tracking/getup motions stay reproducible.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from src import SRC_PATH
from src.assets.robots.wbot import wbot_20dof_constants as _base
from src.assets.robots.wbot.wbot_20dof_constants import (  # noqa: F401
  FEET_ONLY_COLLISION,
  FULL_COLLISION,
  FULL_COLLISION_WITHOUT_SELF,
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
  WBOT_20DOF_ACTION_SCALE,
  WBOT_20DOF_ARTICULATION,
)

WBOT_20DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "wbot2_roban_20260928" / "xmls" / "wbot_20dof.xml"
)
assert WBOT_20DOF_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  # ``xmls/meshes`` is a symlink to the main WBOT mesh pack.
  assets: dict[str, bytes] = {}
  source_dir = (WBOT_20DOF_XML.parent / "meshes").resolve()
  if not source_dir.is_dir():
    raise FileNotFoundError(f"WBOT meshes missing: {source_dir}")
  for file in source_dir.rglob("*"):
    if not file.is_file():
      continue
    rel_path = file.relative_to(source_dir).as_posix()
    data = file.read_bytes()
    assets[rel_path] = data
    if meshdir:
      assets[f"{meshdir}/{rel_path}"] = data
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(WBOT_20DOF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_wbot_20dof_robot_cfg(**kwargs):
  """Roban-pinned robot: same actuators/home as main, frozen MJCF.

  Kwargs forward to ``robots.wbot.get_wbot_20dof_robot_cfg`` (e.g.
  ``collision="getup"``). Defaults match historical getup tracking.
  """
  defaults = dict(
    collision="full",
    home_pose="legacy",
    pd_gains="armature",
  )
  defaults.update(kwargs)
  cfg = _base.get_wbot_20dof_robot_cfg(**defaults)
  cfg.spec_fn = get_spec
  return cfg
