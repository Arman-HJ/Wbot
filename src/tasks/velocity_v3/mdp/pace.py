"""PACE-identified fixed encoder bias for sim2real-aligned training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.adam_sp.pace_params import (
  load_pnd_legs_pace_params,
  resolve_pnd_legs_pace_yaml,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def apply_pace_encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pace_params_yaml: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Set fixed per-joint encoder bias from PACE identified_params YAML."""
  asset: Entity = env.scene[asset_cfg.name]
  yaml_path = resolve_pnd_legs_pace_yaml(pace_params_yaml)
  doc = load_pnd_legs_pace_params(yaml_path)
  bias_by_name = doc.encoder_bias_by_joint()

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_names = tuple(doc.joint_order)
  joint_ids, matched_names = asset.find_joints(joint_names)
  if list(matched_names) != list(joint_names):
    raise RuntimeError(
      f"PACE encoder bias joint mismatch: requested {joint_names}, got {list(matched_names)}"
    )

  joint_ids_tensor = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  bias_values = torch.tensor(
    [bias_by_name[name] for name in matched_names],
    device=env.device,
    dtype=asset.data.encoder_bias.dtype,
  )
  asset.data.encoder_bias[env_ids[:, None], joint_ids_tensor] = bias_values.unsqueeze(0)
