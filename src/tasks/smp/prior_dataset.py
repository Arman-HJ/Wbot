"""Offline env ``disc`` windows for TinyMDM prior training (matches SMP RL)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.tasks.smp.env_disc_replay import (
  build_smp_prior_replay_env,
  collect_env_disc_windows,
)


class EnvDiscPriorDataset:
  """Clip-local windows of env ``disc`` obs (IMU ang_vel + joint pos/vel, H steps)."""

  def __init__(
    self,
    *,
    amp_data_path: str | Path,
    datasets: dict[str, float] | None = None,
    simulation_dt: float,
    num_disc_obs_steps: int = 10,
    slow_down_factor: int = 1,
    device: str = "cpu",
    env_task: str = "AdamSP-SMP-Flat",
  ) -> None:
    del simulation_dt  # env step_dt comes from task cfg (must match motion loader dt)
    self.num_disc_obs_steps = int(num_disc_obs_steps)
    self.device = device
    self.amp_data_path = str(Path(amp_data_path).expanduser().resolve())
    self.env_task = env_task

    self._env, self._library, self._joint_ids = build_smp_prior_replay_env(
      amp_data_path=amp_data_path,
      datasets=datasets,
      slow_down_factor=slow_down_factor,
      device=device,
      env_task=env_task,
    )
    flat, clip_lengths, input_channel = collect_env_disc_windows(
      self._env,
      self._library,
      self._joint_ids,
      history=self.num_disc_obs_steps,
    )
    self.windows = flat.to(device)
    self.clip_lengths = clip_lengths
    self.input_channel = int(input_channel)
    self.input_dim = self.input_channel * self.num_disc_obs_steps
    self._num_windows = int(self.windows.shape[0])

  def close(self) -> None:
    if self._env is not None:
      self._env.close()
      self._env = None

  @property
  def obs_dim(self) -> int:
    return self.input_dim

  def fetch_obs_demo(self, num_samples: int) -> torch.Tensor:
    idx = torch.randint(0, self._num_windows, (num_samples,), device=self.device)
    return self.windows[idx]

  def fetch_obs_demo_batch(self, num_samples: int) -> torch.Tensor:
    flat = self.fetch_obs_demo(num_samples)
    return flat.reshape(num_samples, self.num_disc_obs_steps, self.input_channel)

  def make_eval_indices(self, num_eval: int, seed: int = 42) -> torch.Tensor:
    n = min(int(num_eval), self._num_windows)
    gen = torch.Generator(device=self.device)
    gen.manual_seed(int(seed))
    return torch.randperm(self._num_windows, generator=gen, device=self.device)[:n]

  def gather_windows_hd(self, indices: torch.Tensor) -> torch.Tensor:
    flat = self.windows[indices]
    return flat.reshape(flat.shape[0], self.num_disc_obs_steps, self.input_channel)

  def feature_stats(self) -> dict[str, float]:
    hd = self.windows.reshape(-1, self.num_disc_obs_steps, self.input_channel)
    latest = hd[:, -1, :]
    ang = latest[:, :3]
    j = (self.input_channel - 3) // 2
    jp = latest[:, 3 : 3 + j]
    jv = latest[:, 3 + j :]
    return {
      "ang_vel_mean": float(ang.mean().item()),
      "ang_vel_std": float(ang.std().item()),
      "joint_pos_mean": float(jp.mean().item()),
      "joint_pos_std": float(jp.std().item()),
      "joint_vel_mean": float(jv.mean().item()),
      "joint_vel_std": float(jv.std().item()),
    }

  def summary(self) -> str:
    return (
      f"EnvDiscPriorDataset: task={self.env_task}, clips={len(self.clip_lengths)}, "
      f"lengths={self.clip_lengths}, windows={self._num_windows}, "
      f"channel={self.input_channel}, dim={self.input_dim}, source=env_disc"
    )


# Backward-compatible alias (prior training now always uses env disc).
AmpSmpPriorDataset = EnvDiscPriorDataset
