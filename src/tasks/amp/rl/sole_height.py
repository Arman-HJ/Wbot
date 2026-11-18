"""Play-time thicker-sole control: shift foot collision geoms down in body frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FOOT_COLLISION_GEOMS = r"^(left|right)_foot[1-7]_collision$"
SOLE_HEIGHT_MAX_M = 0.10


class SoleHeightController:
  """Absolute sole height relative to the stock foot collision layout.

  Positive height thickens the sole by subtracting from body-frame Z (same as
  training ``foot_height`` DR). Values are never accumulated across calls.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    asset_cfg = SceneEntityCfg("robot", geom_names=FOOT_COLLISION_GEOMS)
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    entity_indices = _get_entity_indices(
      asset.indexing, asset_cfg, "geom", use_address=False
    ).to(device=env.device, dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
    self._env_grid = env_grid
    self._entity_grid = entity_grid
    self._baseline = env.sim.model.geom_pos[env_grid, entity_grid].clone()
    self._height_m = 0.0

  @property
  def height_m(self) -> float:
    return self._height_m

  def set(self, height_m: float, *, log: bool = False) -> float:
    """Set sole thickness in meters. Clamped to ``[0, SOLE_HEIGHT_MAX_M]``."""
    h = float(max(0.0, min(SOLE_HEIGHT_MAX_M, height_m)))
    pos = self._baseline.clone()
    pos[..., 2] = self._baseline[..., 2] - h
    self._env.sim.model.geom_pos[self._env_grid, self._entity_grid] = pos
    self._height_m = h
    if log:
      print(
        f"[INFO] Sole height +{h * 1000.0:.1f} mm "
        f"(foot collision body-z {-h:+.4f} m)",
        flush=True,
      )
    return h
