"""Play-time ankle_roll-vs-foot offset.

Positive offset shifts foot **collision** and foot **mesh** aft (−X) together
in ``ankle_roll_link``, so ankle_roll sits farther forward relative to the sole
while mesh and contact stay aligned. Pitch mesh is not moved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FOOT_COLLISION_GEOMS = r"^(left|right)_foot[1-7]_collision$"
FOOT_MESH_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ANKLE_ROLL_FORWARD_MAX_M = 0.020


def _foot_mesh_geom_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset = env.scene["robot"]
  body_local, _ = asset.find_bodies(list(FOOT_MESH_BODIES), preserve_order=True)
  target_bodies = {int(asset.indexing.body_ids[i]) for i in body_local}
  mj = env.sim.mj_model
  out: list[int] = []
  for local_gi in range(len(asset.geom_names)):
    gid = int(asset.indexing.geom_ids[local_gi])
    if int(mj.geom_bodyid[gid]) in target_bodies and int(mj.geom_group[gid]) == 2:
      out.append(gid)
  return torch.tensor(out, device=env.device, dtype=torch.long)


class AnkleRollForwardController:
  """Absolute ankle_roll-forward offset relative to the stock foot layout.

  Values are never accumulated across calls; each ``set`` restores from baseline.
  Shifts foot collision capsules and ankle_roll visual meshes aft together.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    asset_cfg = SceneEntityCfg("robot", geom_names=FOOT_COLLISION_GEOMS)
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    collision_indices = _get_entity_indices(
      asset.indexing, asset_cfg, "geom", use_address=False
    ).to(device=env.device, dtype=torch.long)
    mesh_indices = _foot_mesh_geom_ids(env)
    entity_indices = torch.unique(
      torch.cat([collision_indices.reshape(-1), mesh_indices.reshape(-1)])
    )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
    self._env_grid = env_grid
    self._entity_grid = entity_grid
    self._baseline = env.sim.model.geom_pos[env_grid, entity_grid].clone()
    self._offset_m = 0.0
    self._n_collision = int(collision_indices.numel())
    self._n_visual = int(mesh_indices.numel())

  @property
  def offset_m(self) -> float:
    return self._offset_m

  def set(self, offset_m: float, *, log: bool = False) -> float:
    """Set ankle_roll-forward offset in meters. Clamped to ``[0, 20 mm]``."""
    dx = float(max(0.0, min(ANKLE_ROLL_FORWARD_MAX_M, offset_m)))
    pos = self._baseline.clone()
    # Foot (collision + mesh) aft ⇒ ankle_roll relatively forward vs sole.
    pos[..., 0] = self._baseline[..., 0] - dx
    self._env.sim.model.geom_pos[self._env_grid, self._entity_grid] = pos
    self._offset_m = dx
    if log:
      print(
        f"[INFO] Ankle-roll forward +{dx * 1000.0:.1f} mm "
        f"(foot collision+mesh body-x {-dx:+.4f} m; "
        f"n_col={self._n_collision} n_vis={self._n_visual})",
        flush=True,
      )
    return dx
