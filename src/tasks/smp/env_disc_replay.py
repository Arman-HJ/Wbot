"""Replay expert AMP motion in MuJoCo and collect env ``disc`` observation windows."""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401
from src.tasks.amp_v2.mdp.commands import sync_twist_command_from_motion_frames
from src.tasks.amp_v2.mdp.motion_loader import AmpMotionResetLibrary, resolve_amp_datasets


def build_smp_prior_replay_env(
  *,
  amp_data_path: str | Path,
  datasets: dict[str, float] | None,
  slow_down_factor: int,
  device: str,
  env_task: str = "AdamSP-SMP-Flat",
) -> tuple[ManagerBasedRlEnv, AmpMotionResetLibrary, torch.Tensor]:
  """Single-env SMP task configured for expert motion replay."""
  data_dir = Path(amp_data_path).expanduser().resolve()
  dataset_weights = resolve_amp_datasets(data_dir, datasets)

  env_cfg = load_env_cfg(env_task, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  init_motion = env_cfg.events.get("init_amp_motion_library")
  if init_motion is None:
    raise RuntimeError("init_amp_motion_library missing from env cfg")
  init_motion.params["amp_data_path"] = str(data_dir)
  init_motion.params["datasets"] = dataset_weights
  init_motion.params["slow_down_factor"] = int(slow_down_factor)
  for cmd_name in ("expert_ref", "motion"):
    cmd = env_cfg.commands.get(cmd_name)
    if cmd is not None:
      cmd.debug_vis = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  library: AmpMotionResetLibrary = getattr(env, "_amp_motion_reset_library")
  asset = env.scene["robot"]
  joint_asset_cfg = init_motion.params.get("joint_asset_cfg")
  if joint_asset_cfg is not None:
    joint_ids, _ = asset.find_joints(joint_asset_cfg.joint_names, preserve_order=True)
  else:
    joint_ids, _ = asset.find_joints(".*", preserve_order=True)
  return env, library, joint_ids


def apply_motion_frame_to_robot(
  env: ManagerBasedRlEnv,
  library: AmpMotionResetLibrary,
  env_ids: torch.Tensor,
  frame_indices: torch.Tensor,
  joint_ids: slice | torch.Tensor,
) -> None:
  """Write a specific expert frame into the sim (no random sampling)."""
  root_pos, root_quat, joint_pos, joint_vel, root_lin_vel, root_ang_vel = (
    library.gather_reference_states(frame_indices)
  )
  sync_twist_command_from_motion_frames(env, env_ids, frame_indices)
  root_pos = root_pos + env.scene.env_origins[env_ids]

  asset = env.scene["robot"]
  soft_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = torch.clip(joint_pos, soft_limits[..., 0], soft_limits[..., 1])

  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
  default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
  default_joint_pos[:, joint_ids] = joint_pos
  default_joint_vel[:, joint_ids] = joint_vel
  asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

  root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def _advance_frame_and_read_disc(
  env: ManagerBasedRlEnv,
  library: AmpMotionResetLibrary,
  joint_ids: torch.Tensor,
  global_idx: int,
) -> torch.Tensor:
  """Apply one expert frame and return env ``disc`` obs ``(1, H, D)``."""
  device = env.device
  env_ids = torch.tensor([0], device=device, dtype=torch.long)
  frame = torch.tensor([global_idx], device=device, dtype=torch.long)
  apply_motion_frame_to_robot(env, library, env_ids, frame, joint_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  env.command_manager.compute(dt=0.0)
  obs = env.observation_manager.compute(update_history=True)
  return obs["disc"]


@torch.no_grad()
def collect_env_disc_windows(
  env: ManagerBasedRlEnv,
  library: AmpMotionResetLibrary,
  joint_ids: torch.Tensor,
  *,
  history: int,
) -> tuple[torch.Tensor, list[int], int]:
  """Precompute all clip-local ``disc`` windows from MuJoCo replay.

  Returns:
    windows: ``(N, H*D)`` flattened training samples
    clip_lengths: per-clip frame counts from the motion loader
    input_channel: ``D`` (disc feature dim per step)
  """
  loader = library.loader
  h = int(history)
  env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
  windows: list[torch.Tensor] = []

  for clip_i, _name in enumerate(loader.dataset_names):
    start = int(loader.clip_starts[clip_i])
    length = int(loader.clip_lengths[clip_i])
    if length < h:
      continue

    env.observation_manager.reset(env_ids)
    warmup_start = start
    warmup_end = start + h - 1
    for global_idx in range(warmup_start, warmup_end):
      frame = torch.tensor([global_idx], device=env.device, dtype=torch.long)
      apply_motion_frame_to_robot(env, library, env_ids, frame, joint_ids)
      env.scene.write_data_to_sim()
      env.sim.forward()
      env.sim.sense()
      env.observation_manager.compute(update_history=True)

    for global_idx in range(start + h - 1, start + length):
      disc = _advance_frame_and_read_disc(env, library, joint_ids, global_idx)
      windows.append(disc.reshape(-1))

  if not windows:
    raise ValueError(f"No valid env disc windows (history={h})")

  flat = torch.stack(windows, dim=0)
  input_channel = int(flat.shape[-1] // h)
  return flat, [int(x) for x in loader.clip_lengths], input_channel
