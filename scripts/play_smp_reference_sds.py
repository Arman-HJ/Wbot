#!/usr/bin/env python3
"""Play expert AMP reference motion in MuJoCo and show live SMP/SDS scores.

Scores reference motion with env ``disc`` obs — same path as SMP prior training and RL.

Example:
  python scripts/play_smp_reference_sds.py \\
    --motion-file src/assets/amp_motions/pnd_adam_sp/engineai_locomotion/locomotion_pnd_adam_sp.npz

Headless sanity check (no viewer):
  python scripts/play_smp_reference_sds.py --headless-frames 200
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.base import VerbosityLevel

import src.tasks  # noqa: F401
from src.tasks.amp_v2.mdp.motion_loader import AmpMotionResetLibrary
from src.tasks.smp.env_disc_replay import apply_motion_frame_to_robot
from src.tasks.smp.tinymdm.tinymdm_model import TinyMDMModel
from src.viewer.stable_camera_viewer import StableCameraNativeViewer
from rsl_rl.utils.diff_normalizer import DiffNormalizer
from rsl_rl.utils import resolve_callable


@dataclass(frozen=True)
class PlayReferenceSdsConfig:
  motion_file: str = (
    "src/assets/amp_motions/pnd_adam_sp/engineai_locomotion/locomotion_pnd_adam_sp.npz"
  )
  prior_cfg: str = "logs/smp_prior/engineai_prior_env_100k/diffusion_config.yaml"
  prior_model: str = "logs/smp_prior/engineai_prior_env_100k/model_99999.pt"
  sds_loss_scale: float = 6.0
  smp_reward_scale: float = 1.0
  diffusion_steps: tuple[int, ...] = (22, 15, 8)
  num_disc_obs_steps: int = 10
  print_hz: float = 5.0
  stable_camera: bool = True
  warmup_samples: int = 4096
  """Random motion windows to warm up SDS normalizer before playback."""
  headless_frames: int | None = None
  """If set, run N frames without MuJoCo viewer (for CI / quick checks)."""
  device: str | None = None


class ZeroPolicy:
  def __call__(self, obs) -> torch.Tensor:
    del obs
    env = PlayReferenceSdsViewer.env_ref
    return torch.zeros(
      (env.num_envs, env.unwrapped.action_manager.total_action_dim),
      device=env.unwrapped.device,
    )


class SdsScorer:
  """Frozen TinyMDM + DiffNormalizer SDS (same as ``PPOSMP._calc_smp_rewards``)."""

  def __init__(self, cfg: PlayReferenceSdsConfig, disc_obs_dim: int, device: str) -> None:
    self.device = device
    self.sds_loss_scale = float(cfg.sds_loss_scale)
    self.smp_reward_scale = float(cfg.smp_reward_scale)
    self.diffusion_steps = list(cfg.diffusion_steps)
    self.disc_obs_steps = int(cfg.num_disc_obs_steps)
    self._sds_normalizer = DiffNormalizer(
      [len(self.diffusion_steps)], device=device, dtype=torch.float32
    )
    self._prior = self._load_prior(cfg, disc_obs_dim, device)

  def _load_prior(self, cfg: PlayReferenceSdsConfig, disc_obs_dim: int, device: str):
    cfg_path = Path(cfg.prior_cfg).expanduser().resolve()
    model_path = Path(cfg.prior_model).expanduser().resolve()
    if not cfg_path.is_file():
      raise FileNotFoundError(f"Missing prior config: {cfg_path}")
    if not model_path.is_file():
      raise FileNotFoundError(f"Missing prior model: {model_path}")

    with open(cfg_path, "r") as f:
      prior_config = yaml.safe_load(f)
    prior_config["device"] = device
    prior_config["input_dim"] = disc_obs_dim * self.disc_obs_steps
    prior_config["num_disc_obs_steps"] = self.disc_obs_steps
    if "input_channel" not in prior_config:
      prior_config["input_channel"] = disc_obs_dim

    prior_cls = resolve_callable("src.tasks.smp.tinymdm.tinymdm_model:TinyMDMModel")
    model: TinyMDMModel = prior_cls(prior_config)
    state = torch.load(model_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible[0] or incompatible[1]:
      print(f"[WARN] prior load incompatible_keys: {incompatible}")
    model.eval()
    model.to(device)
    for p in model.parameters():
      p.requires_grad = False
    print(f"[INFO] Loaded SMP prior: {model_path}")
    return model

  @torch.no_grad()
  def score_disc(self, disc_obs: torch.Tensor, *, record: bool = True) -> dict[str, float]:
    """``disc_obs``: (N, T, D) → metrics dict."""
    norm = self._prior.normalize(disc_obs)
    flat = norm.reshape(disc_obs.shape[0], -1)
    sds_losses = self._prior.ESM_SDS_loss(norm_x_obs=flat, t_lst=self.diffusion_steps)
    raw_mean = torch.mean(sds_losses, dim=-1)
    if record:
      self._sds_normalizer.record(sds_losses)
      self._sds_normalizer.update()
    sds_norm = self._sds_normalizer.normalize(sds_losses)
    mean_sds_norm = torch.mean(sds_norm, dim=-1)
    smp_r = torch.exp(-mean_sds_norm * self.sds_loss_scale) * self.smp_reward_scale
    return {
      "sds_loss_raw": float(raw_mean.mean().item()),
      "sds_loss_norm_mean": float(mean_sds_norm.mean().item()),
      "smp_reward": float(smp_r.mean().item()),
    }


def warmup_sds_normalizer_env(
  scorer: SdsScorer,
  env: ManagerBasedRlEnv,
  library: AmpMotionResetLibrary,
  joint_ids: torch.Tensor,
  frame_start: int,
  frame_end: int,
  history: int,
  num_samples: int,
) -> None:
  if num_samples <= 0 or frame_end <= frame_start:
    return
  uenv = env.unwrapped
  device = uenv.device
  env_ids = torch.tensor([0], device=device, dtype=torch.long)
  frames = torch.randint(frame_start, frame_end, (num_samples,), device=device)
  uenv.observation_manager.reset(env_ids)
  warmup0 = max(0, int(frames[0].item()) - (history - 1))
  for f in range(warmup0, int(frames[0].item())):
    idx = torch.tensor([f], device=device, dtype=torch.long)
    apply_motion_frame_to_robot(uenv, library, env_ids, idx, joint_ids)
    uenv.scene.write_data_to_sim()
    uenv.sim.forward()
    uenv.sim.sense()
    uenv.observation_manager.compute(update_history=True)
  batch: list[torch.Tensor] = []
  for f in frames:
    idx = torch.tensor([int(f.item())], device=device, dtype=torch.long)
    apply_motion_frame_to_robot(uenv, library, env_ids, idx, joint_ids)
    uenv.scene.write_data_to_sim()
    uenv.sim.forward()
    uenv.sim.sense()
    uenv.command_manager.compute(dt=0.0)
    obs = uenv.observation_manager.compute(update_history=True)
    batch.append(obs["disc"])
  disc = torch.cat(batch, dim=0)
  scorer.score_disc(disc, record=True)
  print(f"[INFO] SDS normalizer warmed up on {num_samples} env-disc windows.")


def _resolve_clip_range(
  library: AmpMotionResetLibrary, clip_stem: str, history: int
) -> tuple[int, int]:
  loader = library.loader
  if clip_stem not in loader.dataset_names:
    raise ValueError(
      f"Clip '{clip_stem}' not in loader. Available: {loader.dataset_names}"
    )
  clip_i = loader.dataset_names.index(clip_stem)
  start = int(loader.clip_starts[clip_i])
  length = int(loader.clip_lengths[clip_i])
  return start + history - 1, start + length


class PlayReferenceSdsViewer(StableCameraNativeViewer):
  env_ref = None

  def __init__(
    self,
    env,
    scorer: SdsScorer,
    library: AmpMotionResetLibrary,
    clip_stem: str,
    frame_start: int,
    frame_end: int,
    joint_ids: torch.Tensor,
    history: int,
    print_hz: float,
  ) -> None:
    self._scorer = scorer
    self._library = library
    self._clip_stem = clip_stem
    self._frame = int(frame_start)
    self._frame_start = int(frame_start)
    self._frame_end = int(frame_end)
    self._joint_ids = joint_ids
    self._history = history
    self._print_period = 1.0 / max(print_hz, 0.1)
    self._last_print = 0.0
    self._last_metrics: dict[str, float] = {}
    type(self).env_ref = env
    super().__init__(env, ZeroPolicy(), verbosity=VerbosityLevel.INFO)

  def setup(self) -> None:
    super().setup()
    self._warmup_disc_history()

  def _warmup_disc_history(self) -> None:
    uenv = self.env.unwrapped
    env_ids = torch.tensor([0], device=uenv.device, dtype=torch.long)
    warmup_start = max(0, self._frame_start - (self._history - 1))
    for f in range(warmup_start, self._frame_start):
      idx = torch.tensor([f], device=uenv.device, dtype=torch.long)
      apply_motion_frame_to_robot(uenv, self._library, env_ids, idx, self._joint_ids)
      uenv.scene.write_data_to_sim()
      uenv.sim.forward()
      uenv.sim.sense()
      uenv.observation_manager.compute(update_history=True)

  def _push_reference_frame(self, global_idx: int) -> dict[str, float]:
    uenv = self.env.unwrapped
    env_ids = torch.tensor([0], device=uenv.device, dtype=torch.long)
    frame = torch.tensor([global_idx], device=uenv.device, dtype=torch.long)
    apply_motion_frame_to_robot(uenv, self._library, env_ids, frame, self._joint_ids)
    uenv.scene.write_data_to_sim()
    uenv.sim.forward()
    uenv.sim.sense()
    uenv.command_manager.compute(dt=0.0)
    obs = uenv.observation_manager.compute(update_history=True)

    disc_env = obs["disc"]
    m_env = self._scorer.score_disc(disc_env, record=False)

    local = self._library.global_to_clip_local(frame, self._clip_stem)
    metrics = {
      "frame": float(global_idx),
      "clip_local": float(local.item()),
      "sds_raw": m_env["sds_loss_raw"],
      "sds_norm": m_env["sds_loss_norm_mean"],
      "smp_r": m_env["smp_reward"],
    }
    self._last_metrics = metrics
    return metrics

  def _execute_step(self) -> bool:
    try:
      with torch.no_grad():
        metrics = self._push_reference_frame(self._frame)
        now = time.perf_counter()
        if now - self._last_print >= self._print_period:
          self._last_print = now
          print(
            f"[SDS] f={int(metrics['frame']):4d} "
            f"loc={int(metrics['clip_local']):4d} | "
            f"sds_raw={metrics['sds_raw']:.2f} "
            f"smp_r={metrics['smp_r']:.3f}",
            flush=True,
          )
        self._frame += 1
        if self._frame >= self._frame_end:
          self._frame = self._frame_start
        self._step_count += 1
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      print(f"[ERROR] {self._last_error}", flush=True)
      self.pause()
      return False

  def _set_status_overlay(self, viewer) -> None:
    import mujoco

    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    m = self._last_metrics
    sds_line = (
      f"sds={m['sds_raw']:.2f}  smp_r={m['smp_r']:.2f}"
      if m
      else "warming up..."
    )
    text_1 = "Env\nStep\nStatus\nSpeed\nSDS / smp_r"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{sds_line}"
    )
    viewer.set_texts(
      (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        text_1,
        text_2,
      )
    )


from src.tasks.smp.env_disc_replay import build_smp_prior_replay_env


def _build_env(cfg: PlayReferenceSdsConfig, clip_stem: str, data_dir: Path, device: str):
  env, _library, _joint_ids = build_smp_prior_replay_env(
    amp_data_path=data_dir,
    datasets={clip_stem: 1.0},
    slow_down_factor=1,
    device=device,
  )
  return env


def run_headless(
  env: ManagerBasedRlEnv,
  scorer: SdsScorer,
  library: AmpMotionResetLibrary,
  clip_stem: str,
  frame_start: int,
  frame_end: int,
  joint_ids: torch.Tensor,
  history: int,
  num_frames: int,
  print_hz: float,
) -> None:
  viewer = PlayReferenceSdsViewer(
    env,
    scorer,
    library,
    clip_stem,
    frame_start,
    frame_end,
    joint_ids,
    history,
    print_hz,
  )
  viewer._warmup_disc_history()
  rewards: list[float] = []
  period = 1.0 / max(print_hz, 0.1)
  last = 0.0
  frame = frame_start
  for _ in range(num_frames):
    metrics = viewer._push_reference_frame(frame)
    rewards.append(metrics["smp_r"])
    now = time.perf_counter()
    if now - last >= period:
      last = now
      print(
        f"[SDS] f={int(metrics['frame']):4d} | "
        f"sds_raw={metrics['sds_raw']:.2f} "
        f"smp_r={metrics['smp_r']:.3f}",
        flush=True,
      )
    frame += 1
    if frame >= frame_end:
      frame = frame_start
  print(
    f"[SUMMARY] {num_frames} frames | smp_r(env) mean={sum(rewards)/len(rewards):.4f}"
  )


def run(cfg: PlayReferenceSdsConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not motion_path.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  clip_stem = motion_path.stem
  data_dir = motion_path.parent

  env = _build_env(cfg, clip_stem, data_dir, device)
  library: AmpMotionResetLibrary = getattr(env, "_amp_motion_reset_library")
  joint_ids, _ = env.scene["robot"].find_joints(".*", preserve_order=True)

  group_dim = env.observation_manager.group_obs_dim["disc"]
  if isinstance(group_dim, tuple) and len(group_dim) == 2:
    history = int(group_dim[0])
    disc_dim = int(group_dim[1])
  else:
    history = cfg.num_disc_obs_steps
    disc_dim = int(group_dim[0]) // history
  frame_start, frame_end = _resolve_clip_range(library, clip_stem, history)
  scorer = SdsScorer(cfg, disc_obs_dim=disc_dim, device=device)
  warmup_sds_normalizer_env(
    scorer,
    env,
    library,
    joint_ids,
    frame_start,
    frame_end,
    history,
    cfg.warmup_samples,
  )

  print(f"[INFO] Clip '{clip_stem}' frames [{frame_start}, {frame_end})")
  print(f"[INFO] disc dim={disc_dim}, history={history}, device={device}")
  print("[INFO] scoring env disc obs only (same as prior training / SMP RL).")

  try:
    if cfg.headless_frames is not None:
      run_headless(
        env,
        scorer,
        library,
        clip_stem,
        frame_start,
        frame_end,
        joint_ids,
        history,
        cfg.headless_frames,
        cfg.print_hz,
      )
    else:
      print("[INFO] Close the MuJoCo window to exit.")
      viewer = PlayReferenceSdsViewer(
        env,
        scorer,
        library,
        clip_stem,
        frame_start,
        frame_end,
        joint_ids,
        history,
        cfg.print_hz,
      )
      viewer.run()
  finally:
    env.close()


def main() -> None:
  cfg = tyro.cli(
    PlayReferenceSdsConfig,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.UsePythonSyntaxForLiteralCollections),
  )
  run(cfg)


if __name__ == "__main__":
  main()
