"""FALCON-style upper-body reference motion (joint tracking, not AMP pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Same 14-DoF arm order as FALCON ``ref_upper_dof_pos`` (waist excluded).
DEFAULT_UPPER_ARM_JOINT_NAMES: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)


@dataclass
class _MotionClip:
  name: str
  fps: float
  joint_pos: torch.Tensor  # (T, D)


class UpperBodyMotionLibrary:
  """Load FALCON ``motions/`` packs (``playlist.yaml`` + ``*_upper14.npy``)."""

  def __init__(
    self,
    motion_dir: str | Path,
    *,
    joint_names: tuple[str, ...] = DEFAULT_UPPER_ARM_JOINT_NAMES,
    device: str = "cpu",
  ) -> None:
    root = Path(motion_dir).expanduser().resolve()
    playlist_path = root / "playlist.yaml"
    if not playlist_path.is_file():
      raise FileNotFoundError(f"Upper-body motion playlist not found: {playlist_path}")

    with open(playlist_path, encoding="utf-8") as handle:
      playlist = yaml.safe_load(handle) or {}

    clips_cfg = playlist.get("clips") or []
    if not clips_cfg:
      raise ValueError(f"Empty upper-body motion playlist: {playlist_path}")

    self.clips: list[_MotionClip] = []
    for entry in clips_cfg:
      upper_file = entry.get("upper14") or entry.get("upper_dof")
      if not upper_file:
        raise KeyError(f"Clip missing upper14 key: {entry}")
      path = root / str(upper_file)
      if not path.is_file():
        raise FileNotFoundError(f"Upper-body clip not found: {path}")
      arr = np.load(path).astype(np.float32)
      if arr.ndim != 2:
        raise ValueError(f"Expected 2D upper motion in {path}, got shape {arr.shape}")
      if arr.shape[1] != len(joint_names):
        raise ValueError(
          f"{path}: width {arr.shape[1]} != expected {len(joint_names)} arm DoF"
        )
      self.clips.append(
        _MotionClip(
          name=str(entry.get("slug") or upper_file),
          fps=float(entry.get("fps", 30.0)),
          joint_pos=torch.tensor(arr, dtype=torch.float32, device=device),
        )
      )

    self.num_clips = len(self.clips)
    self.joint_names = joint_names

  def clip_length_s(self, clip_id: torch.Tensor) -> torch.Tensor:
    lengths = torch.tensor(
      [max(int(c.joint_pos.shape[0]), 1) / max(c.fps, 1e-6) for c in self.clips],
      device=clip_id.device,
      dtype=torch.float32,
    )
    return lengths[clip_id]

  def sample(
    self,
    clip_ids: torch.Tensor,
    times_s: torch.Tensor,
  ) -> torch.Tensor:
    """Linear interpolation; returns absolute joint positions ``(N, D)``."""
    out = torch.zeros(
      clip_ids.shape[0],
      len(self.joint_names),
      device=clip_ids.device,
      dtype=torch.float32,
    )
    for clip_idx, clip in enumerate(self.clips):
      mask = clip_ids == clip_idx
      if not mask.any():
        continue
      t = times_s[mask]
      fps = max(float(clip.fps), 1e-6)
      frame = t * fps
      t_max = max(int(clip.joint_pos.shape[0]) - 1, 0)
      frame = torch.clamp(frame, 0.0, float(t_max))
      i0 = torch.floor(frame).long()
      i1 = torch.clamp(i0 + 1, max=t_max)
      alpha = (frame - i0.float()).unsqueeze(-1)
      q0 = clip.joint_pos[i0]
      q1 = clip.joint_pos[i1]
      out[mask] = (1.0 - alpha) * q0 + alpha * q1
    return out


class UpperBodyMotionCommand(CommandTerm):
  """Play upper-body reference joint trajectories (FALCON ``ref_upper_dof_pos``)."""

  cfg: UpperBodyMotionCommandCfg

  def __init__(self, cfg: UpperBodyMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._enabled = bool(cfg.motion_dir)
    self.library: UpperBodyMotionLibrary | None = None
    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    self.fix_upper_body = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.ref_joint_pos = torch.zeros(
      self.num_envs, len(cfg.joint_names), dtype=torch.float32, device=self.device
    )
    self._default_upper = torch.zeros(
      self.num_envs, len(cfg.joint_names), dtype=torch.float32, device=self.device
    )

    joint_ids, matched = self.robot.find_joints(
      list(cfg.joint_names), preserve_order=True
    )
    missing = [n for n in cfg.joint_names if n not in matched]
    if missing:
      raise ValueError(f"Upper-body motion joints missing on robot: {missing}")
    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    self._default_upper[:] = self.robot.data.default_joint_pos[:, self._joint_ids]

    if self._enabled:
      self.library = UpperBodyMotionLibrary(
        cfg.motion_dir,
        joint_names=cfg.joint_names,
        device=self.device,
      )
      assert self.library.num_clips > 0

    self.metrics["clip_id"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["motion_time"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["fix_upper_body"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.ref_joint_pos

  def _sample_fix_mask(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    prob = float(self.cfg.fix_upper_body_prob)
    self.fix_upper_body[env_ids] = torch.rand(n, device=self.device) < prob

  def _assign_clip_and_time(self, env_ids: torch.Tensor) -> None:
    if not self._enabled or self.library is None or len(env_ids) == 0:
      return
    n = len(env_ids)
    self.clip_ids[env_ids] = torch.randint(
      0, self.library.num_clips, (n,), device=self.device
    )
    lengths = self.library.clip_length_s(self.clip_ids[env_ids])
    self.motion_times[env_ids] = torch.rand(n, device=self.device) * lengths

  def _refresh_reference(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if not self._enabled or self.library is None:
      self.ref_joint_pos[env_ids] = self._default_upper[env_ids]
      return

    moving = env_ids[~self.fix_upper_body[env_ids]]
    fixed = env_ids[self.fix_upper_body[env_ids]]
    if len(moving) > 0:
      self.ref_joint_pos[moving] = self.library.sample(
        self.clip_ids[moving], self.motion_times[moving]
      )
    if len(fixed) > 0:
      self.ref_joint_pos[fixed] = self._default_upper[fixed]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._sample_fix_mask(env_ids)
    self._assign_clip_and_time(env_ids)
    self._refresh_reference(env_ids)

  def _update_command(self) -> None:
    if not self._enabled or self.library is None:
      return
    dt = self._env.step_dt
    self.motion_times += dt
    lengths = self.library.clip_length_s(self.clip_ids)
    wrapped = self.motion_times >= lengths
    if wrapped.any():
      wrap_ids = wrapped.nonzero(as_tuple=False).flatten()
      self._assign_clip_and_time(wrap_ids)
    self._refresh_reference()

  def _update_metrics(self) -> None:
    self.metrics["clip_id"][:] = self.clip_ids.float()
    self.metrics["motion_time"][:] = self.motion_times
    self.metrics["fix_upper_body"][:] = self.fix_upper_body.float()


@dataclass(kw_only=True)
class UpperBodyMotionCommandCfg(CommandTermCfg):
  """Configuration for FALCON-style upper-body joint reference playback."""

  entity_name: str = "robot"
  motion_dir: str = ""
  """Directory with ``playlist.yaml`` and ``*_upper14.npy`` clips."""
  joint_names: tuple[str, ...] = DEFAULT_UPPER_ARM_JOINT_NAMES
  fix_upper_body_prob: float = 0.3
  """Fraction of envs that hold default arm pose instead of playing motion."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyMotionCommand:
    return UpperBodyMotionCommand(self, env)
