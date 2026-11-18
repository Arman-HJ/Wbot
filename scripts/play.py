"""Script to play RL agent with RSL-RL."""

import os
import re
import shutil
import signal
import sys
from dataclasses import asdict, dataclass, replace
from numbers import Number
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro


def _copy_motion_file_to_log(motion_file: str | Path, log_dir: Path) -> Path:
  """Copy motion npz into the run log dir; return the destination path."""
  src = Path(motion_file).expanduser().resolve()
  if not src.is_file():
    raise FileNotFoundError(f"Motion file not found: {src}")
  log_dir.mkdir(parents=True, exist_ok=True)
  dst = log_dir / src.name
  if src != dst:
    shutil.copy2(src, dst)
  print(f"[INFO] Copied motion file -> {dst}")
  return dst

import mjlab

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from run_path import resolve_run
from src.env_scale_print import print_env_input_output_scales, print_robot_init_and_defaults
from src.viewer.live_state_printer import wrap_viewer_with_live_state_print
from src.viewer.stable_camera_viewer import StableCameraNativeViewer

# mjlab.TYRO_FLAGS disables ``--flag`` bool syntax (requires ``--flag True``).
# Play uses a separate config so ``--no-stable-camera`` works as a flag.
PLAY_TYRO_FLAGS = (
  tyro.conf.AvoidSubcommands,
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  run: str | None = None
  """Run name under logs/rsl_rl/<experiment>/ (wasm: whole run; else ``run/model_X.pt``);
  a full path also works. Else set `wandb_run_path`."""
  wandb_run_path: str | None = None
  """WandB run path (e.g. entity/project/run_id) for latest checkpoint when no local file."""
  registry_name: str | None = None
  """WandB motions registry for tracking dummy mode without `--motion-file`."""
  motion_file: str | None = None
  amp_vis_motion: str | None = None
  """Expert ``.npz`` for AMP expert ghost (``expert_ref`` command)."""
  amp_data_path: str | None = None
  """Optional expert motion dir (training / ghost). Not required for AMP v2 play."""
  amp_datasets: str | None = None
  """Optional ``name:weight`` pairs when loading expert clips (train or optional ghost)."""
  slow_down_factor: int = 1
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 100
  """Headless --video: record this many frames then exit automatically."""
  video_height: int | None = None
  video_width: int | None = None
  video_output: str | None = None
  """Output mp4 path or directory (--video headless mode). Default: <ckpt_dir>/videos/play/."""
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "wasm"] = "auto"
  stable_camera: bool = True
  """Native viewer: FREE cam + follow torso (avoids TRACKING subtree_com jitter)."""
  viser_host: str = "0.0.0.0"
  """Viser / wasm bind address. Use 0.0.0.0 so Mac/LAN browsers can connect without SSH -L."""
  viser_port: int = 8080
  """Viser HTTP/WebSocket port, or Vite port when ``--viewer wasm``."""
  viser_share: bool = False
  """Open a public Cloudflare quick tunnel (no SSH port-forward). Prints https://….trycloudflare.com."""
  print_live_state: bool = True
  """Print sum of |actuator torque| to the console during play."""
  print_live_state_hz: float = 5.0
  """Console print rate for ``print_live_state`` (Hz)."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  lin_vel_x: float | None = None
  """Velocity tasks: fixed forward command in m/s (play cfg defaults to 0)."""
  lin_vel_y: float | None = None
  """Velocity tasks: fixed lateral command in m/s."""
  ang_vel_z: float | None = None
  """Velocity tasks: fixed yaw-rate command in rad/s."""
  export_on_load: bool = True
  """In trained mode, call runner.save once after loading to trigger default exports (e.g. ONNX)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _lan_ipv4_addresses() -> list[str]:
  """Non-loopback IPv4 addresses for printing LAN Viser URLs."""
  import socket

  addrs: list[str] = []
  try:
    from subprocess import check_output

    for token in check_output(["hostname", "-I"], text=True).split():
      if token and not token.startswith("127."):
        addrs.append(token)
  except Exception:  # noqa: BLE001
    pass
  if not addrs:
    try:
      hostname = socket.gethostname()
      for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
        ip = info[4][0]
        if ip and not ip.startswith("127.") and ip not in addrs:
          addrs.append(ip)
    except Exception:  # noqa: BLE001
      pass
  return addrs


def _print_viser_access_urls(host: str, port: int) -> None:
  print(f"[INFO] Viser listening on {host}:{port}", flush=True)
  print(f"[INFO] Local:  http://127.0.0.1:{port}/", flush=True)
  for ip in _lan_ipv4_addresses():
    print(f"[INFO] LAN:    http://{ip}:{port}/", flush=True)
  print(
    "[INFO] If Mac browser shows ERR_ADDRESS_UNREACHABLE to the LAN URL while "
    "SSH works: System Settings → Privacy & Security → Local Network → enable "
    "Chrome/Safari for this Mac.",
    flush=True,
  )


def _start_cloudflared_share(port: int):
  """Start Cloudflare quick tunnel; return Popen or None."""
  import shutil
  import subprocess
  from pathlib import Path

  candidates = [
    shutil.which("cloudflared"),
    str(Path.home() / ".local" / "bin" / "cloudflared"),
  ]
  binary = next((c for c in candidates if c and Path(c).is_file()), None)
  if binary is None:
    print(
      "[WARN] --viser-share needs cloudflared. "
      "Install: curl -L -o ~/.local/bin/cloudflared "
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
      "&& chmod +x ~/.local/bin/cloudflared",
      flush=True,
    )
    return None

  proc = subprocess.Popen(
    [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )

  def _pump() -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
      line = line.rstrip()
      print(f"[cloudflared] {line}", flush=True)
      if "trycloudflare.com" in line and "https://" in line:
        # Extract first https URL on the line.
        for part in line.split():
          if part.startswith("https://") and "trycloudflare.com" in part:
            print(f"[INFO] Public Viser URL (no SSH forward): {part}", flush=True)
            break

  import threading

  threading.Thread(target=_pump, daemon=True).start()
  print("[INFO] Starting Cloudflare quick tunnel for Viser…", flush=True)
  return proc


def _resolve_gain(gain, joint_name: str) -> float | None:
  if isinstance(gain, Number):
    return float(gain)
  if isinstance(gain, dict):
    if joint_name in gain and isinstance(gain[joint_name], Number):
      return float(gain[joint_name])
    for expr, value in gain.items():
      if isinstance(value, Number) and re.fullmatch(expr, joint_name):
        return float(value)
  return None


def _print_joint_pd_values(env_cfg, env: ManagerBasedRlEnv) -> None:
  robot = env.scene["robot"]
  robot_cfg = env_cfg.scene.entities.get("robot")
  articulation = getattr(robot_cfg, "articulation", None)
  actuators = getattr(articulation, "actuators", None)
  if actuators is None:
    print("[INFO] No actuator config found; skip PD print.")
    return

  pd_map: dict[str, tuple[float | None, float | None]] = {}
  for act in actuators:
    target_expr = getattr(act, "target_names_expr", ())
    if not target_expr:
      continue
    _, matched_names = robot.find_joints(target_expr, preserve_order=True)
    for joint_name in matched_names:
      kp = _resolve_gain(getattr(act, "stiffness", None), joint_name)
      kd = _resolve_gain(getattr(act, "damping", None), joint_name)
      pd_map[joint_name] = (kp, kd)

  print("[INFO] Joint PD gains (Kp, Kd):")
  for joint_name in robot.joint_names:
    if joint_name not in pd_map:
      continue
    kp, kd = pd_map[joint_name]
    kp_str = f"{kp:.6f}" if kp is not None else "N/A"
    kd_str = f"{kd:.6f}" if kd is not None else "N/A"
    print(f"  - {joint_name}: Kp={kp_str}, Kd={kd_str}")



def _parse_amp_datasets(amp_datasets: str | None) -> dict[str, float] | None:
  if not amp_datasets:
    return None
  out: dict[str, float] = {}
  for part in amp_datasets.split(","):
    name, _, weight = part.partition(":")
    name = name.strip()
    if not name:
      continue
    out[name] = float(weight.strip() or "1.0")
  return out or None


def _apply_amp_motion_library_overrides(
  env_cfg,
  *,
  amp_data_path: str | None,
  amp_datasets: str | None,
  slow_down_factor: int,
) -> None:
  """Sync startup motion library with CLI AMP dataset args."""
  from src.tasks.amp_v2.mdp.motion_loader import (
    amp_dataset_yaml_path,
    infer_amp_dataset_source,
    print_amp_datasets_summary,
    resolve_amp_datasets,
  )

  init_motion = env_cfg.events.get("init_amp_motion_library")
  if init_motion is None:
    return
  data_dir: Path | None = None
  if amp_data_path:
    data_dir = Path(amp_data_path).expanduser().resolve()
    init_motion.params["amp_data_path"] = str(data_dir)
  elif init_motion.params.get("amp_data_path"):
    data_dir = Path(str(init_motion.params["amp_data_path"])).expanduser().resolve()

  datasets = _parse_amp_datasets(amp_datasets)
  if data_dir is not None:
    if not datasets:
      if amp_data_path:
        datasets = resolve_amp_datasets(data_dir, None)
      else:
        existing = init_motion.params.get("datasets")
        datasets = existing if existing else resolve_amp_datasets(data_dir, None)

  if datasets:
    init_motion.params["datasets"] = datasets
    if data_dir is not None:
      if amp_datasets:
        source = "explicit (--amp-datasets)"
      else:
        yaml_path = amp_dataset_yaml_path(data_dir)
        source = (
          f"yaml ({yaml_path.name})"
          if yaml_path is not None
          else infer_amp_dataset_source(data_dir, None)
        )
      print_amp_datasets_summary(data_dir, datasets, source=source)
  init_motion.params["slow_down_factor"] = slow_down_factor


def _policy_only_play_train_cfg(rsl_cfg: dict, agent_cfg) -> dict:
  """Play: actor/critic only via ``OnPolicyRunner`` + ``PPO`` (no disc / prior)."""
  alg = rsl_cfg["algorithm"]
  return {
    "seed": rsl_cfg.get("seed", 42),
    "device": rsl_cfg.get("device", "cpu"),
    "multi_gpu": None,
    "class_name": "OnPolicyRunner",
    "obs_groups": {
      "policy": ["actor"],
      "critic": ["critic"],
      "actor": ["actor"],
    },
    "actor": rsl_cfg["actor"],
    "critic": rsl_cfg["critic"],
    "algorithm": {
      "class_name": "PPO",
      "gamma": alg["gamma"],
      "lam": alg["lam"],
      "clip_param": alg["clip_param"],
      "entropy_coef": alg["entropy_coef"],
      "value_loss_coef": alg["value_loss_coef"],
      "use_clipped_value_loss": alg["use_clipped_value_loss"],
      "max_grad_norm": alg["max_grad_norm"],
      "schedule": alg["schedule"],
      "desired_kl": alg["desired_kl"],
      "num_learning_epochs": alg["num_learning_epochs"],
      "num_mini_batches": alg["num_mini_batches"],
      "learning_rate": alg["learning_rate"],
      "optimizer": alg["optimizer"],
    },
    "num_steps_per_env": rsl_cfg["num_steps_per_env"],
    "save_interval": rsl_cfg["save_interval"],
    "max_iterations": rsl_cfg["max_iterations"],
    "clip_actions": rsl_cfg.get("clip_actions"),
    "logger": rsl_cfg.get("logger", "tensorboard"),
    "wandb_project": rsl_cfg.get("wandb_project", "mjlab"),
    "experiment_name": getattr(agent_cfg, "experiment_name", "play"),
  }


def _build_play_train_cfg(task_id: str, agent_cfg, runner_cls, cfg: PlayConfig) -> dict:
  """Build runner train dict for checkpoint load (AMP v2 / SMP play: actor-only)."""
  try:
    from src.tasks.amp_v2.rl.rsl_rl_amp_runner import is_amp_v2_runner_cls
    from src.tasks.amp_v2.rl.rl_cfg import AmpV2RunnerCfg, build_rsl_rl_amp_train_cfg
  except ModuleNotFoundError:
    # amp_v2 package optional; kick / tracking play only needs agent cfg.
    del task_id, runner_cls, cfg
    return asdict(agent_cfg)
  from src.tasks.smp.rl.rl_cfg import SmpRunnerCfg, build_rsl_rl_smp_train_cfg
  from src.tasks.smp.rl.smp_runner import is_smp_runner_cls

  amp_datasets = _parse_amp_datasets(cfg.amp_datasets)
  is_smp = is_smp_runner_cls(runner_cls) or isinstance(agent_cfg, SmpRunnerCfg)
  is_amp_v2 = (is_amp_v2_runner_cls(runner_cls) or isinstance(agent_cfg, AmpV2RunnerCfg)) and not is_smp

  if is_smp:
    smp_cfg = build_rsl_rl_smp_train_cfg(
      agent_cfg,
      amp_data_path=cfg.amp_data_path,
      slow_down_factor=cfg.slow_down_factor,
      datasets=amp_datasets,
      inference_only=True,
    )
    return _policy_only_play_train_cfg(smp_cfg, agent_cfg)

  if is_amp_v2:
    amp_cfg = build_rsl_rl_amp_train_cfg(
      agent_cfg,
      amp_data_path=cfg.amp_data_path,
      slow_down_factor=cfg.slow_down_factor,
      datasets=amp_datasets,
      inference_only=True,
    )
    return _policy_only_play_train_cfg(amp_cfg, agent_cfg)
  del task_id
  return asdict(agent_cfg)


def _strip_amp_discriminator_for_play(env_cfg) -> None:
  """Remove discriminator / expert-demo obs and AMP-only commands for play."""
  for key in ("disc", "disc_demo"):
    env_cfg.observations.pop(key, None)
  env_cfg.events.pop("assert_amp_disc_12dof", None)
  for key in ("motion", "expert_ref"):
    env_cfg.commands.pop(key, None)
  from src.tasks.amp_v2.amp_v2_env_cfg import apply_uniform_reset_events

  # Play does not need expert-frame reset / discriminator; use default pose reset.
  apply_uniform_reset_events(env_cfg)
  print("[INFO] Play mode: AMP discriminator / expert demo disabled (policy only).")


def _align_actor_obs_from_checkpoint(env_cfg, resume_path: Path) -> None:
  """Match actor obs layout to the checkpoint run (history / terms)."""
  align_path = resume_path.parent / "params" / "deployment_alignment.yaml"
  actor_terms: set[str] | None = None
  history_length: int | None = None

  if align_path.is_file():
    try:
      import yaml

      data = yaml.safe_load(align_path.read_text())
    except Exception as exc:  # noqa: BLE001
      print(f"[WARN] Could not parse {align_path.name}: {exc}")
      data = None
    if isinstance(data, dict):
      actor = (data.get("observations") or {}).get("actor") or {}
      history_length = actor.get("history_length")
      terms = actor.get("terms")
      layout = actor.get("layout")
      if isinstance(terms, dict):
        actor_terms = set(terms.keys())
      elif isinstance(layout, list):
        actor_terms = {item["name"] for item in layout if isinstance(item, dict) and "name" in item}
      concat = actor.get("concatenated_dim")
      if concat is not None:
        print(
          f"[INFO] Aligning play obs from {align_path.parent.parent.name}/"
          f"deployment_alignment.yaml (actor_dim={concat}, history={history_length})."
        )

  if history_length is None or actor_terms is None:
    try:
      ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
      actor_in = int(ckpt["actor_state_dict"]["mlp.0.weight"].shape[1])
      actor_out = int(ckpt["actor_state_dict"]["mlp.6.weight"].shape[0])
    except Exception as exc:  # noqa: BLE001
      print(f"[WARN] Could not inspect checkpoint obs size: {exc}")
      return
    # Heuristic fallbacks for known Adam SP AMP layouts.
    if actor_out == 29 and actor_in == 96:
      history_length = 1
      actor_terms = {
        "base_ang_vel",
        "projected_gravity",
        "command",
        "joint_pos",
        "joint_vel",
        "actions",
      }
      print(
        f"[INFO] Checkpoint actor is {actor_in}-D / {actor_out}-act; "
        "using history_length=1 without phase."
      )
    elif actor_out == 12 and actor_in == 47:
      raise RuntimeError(
        "This checkpoint is 12-DoF (obs=47, act=12). "
        "Use task AdamSP-AMP-v2-12DoF-Flat / Rough instead of full-body Flat."
      )
    else:
      print(
        f"[INFO] Checkpoint actor in={actor_in} out={actor_out}; "
        "keeping current env observation layout."
      )
      return

  assert history_length is not None and actor_terms is not None
  term_scales: dict[str, float] = {}
  if align_path.is_file():
    try:
      import yaml

      data = yaml.safe_load(align_path.read_text())
      actor = ((data or {}).get("observations") or {}).get("actor") or {}
      for name, term in (actor.get("terms") or {}).items():
        if isinstance(term, dict) and term.get("scale") is not None:
          term_scales[name] = float(term["scale"])
      for item in actor.get("layout") or []:
        if not isinstance(item, dict) or "name" not in item:
          continue
        # layout.scale is often null; prefer terms dict above.
        if item.get("scale") is not None and item["name"] not in term_scales:
          term_scales[item["name"]] = float(item["scale"])
    except Exception:
      pass

  for group_name in ("actor", "critic"):
    group = env_cfg.observations.get(group_name)
    if group is None:
      continue
    group.history_length = int(history_length)
    keep = set(actor_terms)
    if group_name == "critic":
      keep = keep | {"base_lin_vel"}
    for term_name in list(group.terms.keys()):
      if term_name not in keep:
        group.terms.pop(term_name)
      elif term_name in term_scales:
        group.terms[term_name].scale = term_scales[term_name]


def _load_actor_only(runner, resume_path: Path, device: str) -> None:
  """Load policy weights only; skip critic / discriminator / optimizers."""
  checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
  actor_sd = checkpoint.get("actor_state_dict")
  if actor_sd is None:
    raise KeyError(f"Checkpoint missing actor_state_dict: {resume_path}")
  raw_actor = getattr(runner.alg, "_raw_actor", runner.alg.actor)
  raw_actor.load_state_dict(actor_sd, strict=True)
  iter_i = checkpoint.get("iter")
  if iter_i is not None:
    print(f"[INFO] Loaded actor-only from {resume_path.name} (iter={iter_i}).")
  else:
    print(f"[INFO] Loaded actor-only from {resume_path.name}.")


def _play_video_camera_distance(num_envs: int, spacing: float) -> float:
  """Single-env play: close orbit; multi-env: wide overview."""
  if num_envs <= 1:
    return 3.2
  return max(10.0, spacing * 6.0)


def _apply_play_video_viewer(env_cfg, cfg: PlayConfig) -> None:
  """WORLD overview camera for headless mp4 export."""
  from mjlab.viewer import ViewerConfig

  num_envs = int(env_cfg.scene.num_envs)
  viewer = env_cfg.viewer
  viewer.width = int(cfg.video_width or 1280)
  viewer.height = int(cfg.video_height or 720)
  viewer.origin_type = ViewerConfig.OriginType.WORLD
  viewer.body_name = None
  viewer.fovy = 40.0
  viewer.max_extra_envs = min(127, max(0, num_envs - 1))
  spacing = float(env_cfg.scene.env_spacing)
  viewer.distance = _play_video_camera_distance(num_envs, spacing)
  viewer.elevation = -20.0 if num_envs <= 1 else -25.0
  viewer.azimuth = 135.0
  print(
    "[INFO] Video export camera: WORLD overview "
    f"({viewer.width}x{viewer.height}, max_extra_envs={viewer.max_extra_envs}, "
    f"distance≈{viewer.distance}).",
    flush=True,
  )


def _configure_play_video_camera(env: ManagerBasedRlEnv) -> None:
  renderer = env._offline_renderer
  if renderer is None:
    return

  origins = env.scene.env_origins[: env.num_envs].detach().cpu().numpy()
  spacing = float(env.cfg.scene.env_spacing)
  centroid = origins.mean(axis=0)
  dist2 = np.sum((origins[:, :2] - centroid[:2]) ** 2, axis=1)
  env_idx = int(np.argmin(dist2))
  lookat_z = 0.55 if env.num_envs <= 1 else 0.9
  lookat = (float(origins[env_idx, 0]), float(origins[env_idx, 1]), lookat_z)
  distance = _play_video_camera_distance(int(env.num_envs), spacing)

  viewer = env.cfg.viewer
  viewer.env_idx = env_idx
  viewer.lookat = lookat
  viewer.distance = distance
  renderer._cfg.env_idx = env_idx
  renderer._cfg.lookat = lookat
  renderer._cfg.distance = distance
  renderer._cam.lookat[:] = lookat
  renderer._cam.distance = distance
  renderer._extra_env_ids = None
  renderer._render_extent = renderer._compute_render_extent()
  renderer._model.stat.extent = renderer._render_extent


def _resolve_video_folder(cfg: PlayConfig, log_dir: Path, resume_path: Path) -> Path:
  if cfg.video_output is None:
    return log_dir / "videos" / "play"
  out = Path(cfg.video_output).expanduser()
  if out.suffix.lower() == ".mp4":
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.parent
  out.mkdir(parents=True, exist_ok=True)
  return out


def _find_video_recorder(env) -> VideoRecorder | None:
  cur = env
  while cur is not None:
    if isinstance(cur, VideoRecorder):
      return cur
    cur = getattr(cur, "env", None)
  return None


def _maybe_start_play_clip(env) -> None:
  """Kick / non-loop tracking: advance the reference once (same as Viser Play Clip)."""
  base = env.unwrapped if hasattr(env, "unwrapped") else env
  cmd_mgr = getattr(base, "command_manager", None)
  if cmd_mgr is None:
    return
  try:
    cmd = cmd_mgr.get_term("motion")
  except Exception:
    return
  start = getattr(cmd, "start_play_clip", None)
  if not callable(start):
    return
  ok = bool(start())
  print(f"[INFO] Headless video: start_play_clip -> {ok}", flush=True)


def _run_headless_video(env, policy, video_length: int) -> None:
  recorder = _find_video_recorder(env)
  interrupted = False

  def _handle_sigint(signum, frame) -> None:
    del signum, frame
    nonlocal interrupted
    if not interrupted:
      interrupted = True
      print("\n[INFO] Ctrl+C: stopping rollout and saving video...", flush=True)

  prev_handler = signal.signal(signal.SIGINT, _handle_sigint)
  obs, _ = env.reset()
  _maybe_start_play_clip(env)
  # Refresh actor obs after clip align (policy must see frame-0 command).
  get_obs = getattr(env, "get_observations", None)
  if callable(get_obs):
    obs = get_obs()
  try:
    with torch.inference_mode():
      for step in range(video_length):
        if interrupted:
          break
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if recorder is not None and recorder.video_count >= 1:
          print(
            f"[INFO] Video export complete ({video_length} frames).",
            flush=True,
          )
          break
        if (step + 1) % 25 == 0:
          n_frames = len(recorder.current_video_frames) if recorder else step + 1
          print(
            f"[INFO] Recording {n_frames}/{video_length} frames...",
            flush=True,
          )
  finally:
    signal.signal(signal.SIGINT, prev_handler)


def run_play(task_id: str, cfg: PlayConfig):
  cfg = replace(cfg, run=resolve_run(task_id, cfg.run))
  if cfg.viewer == "wasm":
    if cfg.run is None:
      raise ValueError("--viewer wasm requires --run (run name, or run/model_*.pt)")
    from src.viewer.web_play import launch_web_play

    launch_web_play(
      motion_file=cfg.motion_file,
      checkpoint_file=cfg.run,
      host=cfg.viser_host,
      port=int(cfg.viser_port),
    )
    return

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  # Playback is unperturbed: retain training disturbances only in train configs.
  push_functions = {
    "push_by_setting_velocity", "apply_body_impulse",
    "apply_external_force_torque", "apply_external_wrench",
  }
  removed_pushes = []
  for name, event in list(env_cfg.events.items()):
    function_name = getattr(event.func, "__name__", "")
    if name.startswith("push") or function_name in push_functions:
      removed_pushes.append(name)
      del env_cfg.events[name]
  print(f"[INFO] Play automatic pushes disabled; removed={removed_pushes}")
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  if any(v is not None for v in (cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z)):
    twist_cfg = env_cfg.commands.get("twist")
    if not isinstance(twist_cfg, UniformVelocityCommandCfg):
      raise ValueError(
        f"--lin-vel-x/--lin-vel-y/--ang-vel-z need a 'twist' velocity command; "
        f"task {task_id!r} has none."
      )
    vx = cfg.lin_vel_x or 0.0
    vy = cfg.lin_vel_y or 0.0
    wz = cfg.ang_vel_z or 0.0
    twist_cfg.ranges.lin_vel_x = (vx, vx)
    twist_cfg.ranges.lin_vel_y = (vy, vy)
    twist_cfg.ranges.ang_vel_z = (wz, wz)
    twist_cfg.rel_standing_envs = 0.0
    twist_cfg.rel_heading_envs = 0.0
    env_cfg.curriculum.pop("command_vel", None)
    print(
      f"[INFO]: Fixed velocity command: vx={vx:.2f} m/s, vy={vy:.2f} m/s, "
      f"wz={wz:.2f} rad/s"
    )

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Explicit CLI override (optional; else env cfg / log-dir snapshot).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name and not (
        motion_cmd.motion_file and Path(motion_cmd.motion_file).exists()
      ):
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  env.commands.motion.motion_file in task cfg\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.run is not None:
      resume_path = Path(cfg.run)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError("`wandb_run_path` is required when `run` is not provided.")
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent
    if is_tracking_task:
      motion_cmd = env_cfg.commands["motion"]
      assert isinstance(motion_cmd, MotionCommandCfg)
      # Prefer motion already snapshotted in this run's log dir.
      if cfg.motion_file is None:
        env_name = (
          Path(motion_cmd.motion_file).name if motion_cmd.motion_file else None
        )
        logged = None
        if env_name and (log_dir / env_name).is_file():
          logged = log_dir / env_name
        else:
          npzs = sorted(log_dir.glob("*.npz"))
          if len(npzs) == 1:
            logged = npzs[0]
        if logged is not None:
          motion_cmd.motion_file = str(logged.resolve())
          print(f"[INFO]: Using motion file from log dir: {motion_cmd.motion_file}")
        elif motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
          print(
            f"[INFO]: Using motion file from env cfg: {motion_cmd.motion_file}"
          )
      # Snapshot into log dir if not already there.
      if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
        motion_cmd.motion_file = str(
          _copy_motion_file_to_log(motion_cmd.motion_file, log_dir)
        )

  # AMP v2 / SMP play: policy only — no discriminator / prior at inference.
  play_amp_policy_only = False
  if TRAINED_MODE:
    try:
      from src.tasks.amp_v2.rl.rsl_rl_amp_runner import is_amp_v2_runner_cls
      from src.tasks.amp_v2.rl.rl_cfg import AmpV2RunnerCfg
      from src.tasks.smp.rl.rl_cfg import SmpRunnerCfg
      from src.tasks.smp.rl.smp_runner import is_smp_runner_cls

      runner_cls_peek = load_runner_cls(task_id) or MjlabOnPolicyRunner
      is_smp_task = is_smp_runner_cls(runner_cls_peek) or isinstance(agent_cfg, SmpRunnerCfg)
      is_amp_v2_task = (
        is_amp_v2_runner_cls(runner_cls_peek)
        or (isinstance(agent_cfg, AmpV2RunnerCfg) and not is_smp_task)
      )
      play_amp_policy_only = is_amp_v2_task or is_smp_task
      if play_amp_policy_only:
        _strip_amp_discriminator_for_play(env_cfg)
        if resume_path is not None:
          _align_actor_obs_from_checkpoint(env_cfg, resume_path)
    except ModuleNotFoundError:
      play_amp_policy_only = False

  if "expert_ref" in env_cfg.commands and (cfg.amp_vis_motion or cfg.amp_data_path):
    try:
      from src.tasks.amp_v2.mdp.commands import (
        AmpExpertVisCommandCfg,
        resolve_amp_expert_vis_motion_file,
      )
    except ModuleNotFoundError:
      print(
        "[WARN] expert_ref configured but AMP v2 commands are unavailable; "
        "skipping AMP expert ghost."
      )
    else:
      expert_cmd = env_cfg.commands["expert_ref"]
      if isinstance(expert_cmd, AmpExpertVisCommandCfg):
        vis_name: str | None = None
        if cfg.amp_datasets:
          vis_name = next(iter(cfg.amp_datasets.split(",")), "").partition(":")[0].strip()
        motion_path = resolve_amp_expert_vis_motion_file(
          amp_vis_motion=cfg.amp_vis_motion,
          amp_data_path=cfg.amp_data_path,
          dataset_name=vis_name or None,
        )
        if motion_path:
          expert_cmd.motion_file = motion_path
          expert_cmd.debug_vis = True
          print(f"[INFO] AMP expert ghost motion: {motion_path}")
        else:
          expert_cmd.debug_vis = False
          print(
            "[WARN] AMP expert ghost disabled (no .npz). Pass --amp-vis-motion or "
            "--amp-data-path with expert clips."
          )

  if cfg.amp_data_path:
    _apply_amp_motion_library_overrides(
      env_cfg,
      amp_data_path=cfg.amp_data_path,
      amp_datasets=cfg.amp_datasets,
      slow_down_factor=cfg.slow_down_factor,
    )

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  HEADLESS_VIDEO = TRAINED_MODE and cfg.video
  if HEADLESS_VIDEO:
    # Nominal robot for video: drop startup DR (CoM/friction/PD) so frame-0 stands stable.
    if env_cfg.events:
      dropped = sorted(env_cfg.events.keys())
      env_cfg.events = {}
      print(f"[INFO] Headless video: cleared events {dropped}", flush=True)
    _apply_play_video_viewer(env_cfg, cfg)
    if cfg.num_envs is not None and cfg.num_envs > 32:
      print(
        f"[WARN] Headless video with {cfg.num_envs} envs is slow; "
        "use --num-envs=16 for faster export.",
        flush=True,
      )
    if "expert_ref" in env_cfg.commands:
      try:
        from src.tasks.amp_v2.mdp.commands import AmpExpertVisCommandCfg

        expert_cmd = env_cfg.commands["expert_ref"]
        if isinstance(expert_cmd, AmpExpertVisCommandCfg):
          expert_cmd.debug_vis = False
      except ModuleNotFoundError:
        pass
  else:
    if cfg.video_height is not None:
      env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
      env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if HEADLESS_VIDEO else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  try:
    from src.tasks.amp_v2.rl.rsl_rl_amp_runner import AmpRlEnv, uses_rsl_rl_amp_env
  except ModuleNotFoundError:
    AmpRlEnv = ManagerBasedRlEnv  # type: ignore[misc,assignment]

    def uses_rsl_rl_amp_env(_runner_cls) -> bool:
      del _runner_cls
      return False

  if play_amp_policy_only:
    # Use plain on-policy runner so play never constructs AMPDiscriminator.
    runner_cls = MjlabOnPolicyRunner
    env_cls = ManagerBasedRlEnv
  else:
    env_cls = AmpRlEnv if uses_rsl_rl_amp_env(runner_cls) else ManagerBasedRlEnv
  env = env_cls(cfg=env_cfg, device=device, render_mode=render_mode)
  if not HEADLESS_VIDEO:
    _print_joint_pd_values(env_cfg, env)
    print_robot_init_and_defaults(env_cfg, env)
    print_env_input_output_scales(env_cfg, agent_cfg)

  video_folder: Path | None = None
  if HEADLESS_VIDEO:
    assert log_dir is not None and resume_path is not None
    video_folder = _resolve_video_folder(cfg, log_dir, resume_path)
    print(f"[INFO] Headless video export ({cfg.video_length} frames) -> {video_folder}")
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=False,
    )
    _configure_play_video_camera(env.unwrapped)

  play_train_cfg: dict = asdict(agent_cfg)
  if TRAINED_MODE:
    play_train_cfg = _build_play_train_cfg(task_id, agent_cfg, runner_cls, cfg)
  clip_actions = play_train_cfg.get("clip_actions", agent_cfg.clip_actions)
  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner = runner_cls(env, play_train_cfg, device=device)
    if play_amp_policy_only:
      _load_actor_only(runner, resume_path, device)
    else:
      runner.load(str(resume_path), map_location=device)
    if cfg.export_on_load and not HEADLESS_VIDEO:
      # Prefer task-specific save hook; fallback when logger is not initialized in play mode.
      # Never overwrite the training checkpoint (play often strips disc / resets iter).
      try:
        export_pt = resume_path.with_name(f"{resume_path.stem}.play_export{resume_path.suffix}")
        runner.save(str(export_pt))
        print(f"[INFO]: Exported artifacts by save hook to: {export_pt}")
      except AttributeError as e:
        export_dir = str(Path(resume_path).parent)
        run_name = Path(resume_path).parent.name
        exported_files: list[str] = []
        if hasattr(runner, "export_motion_policy_to_onnx"):
          runner.export_motion_policy_to_onnx(export_dir, f"{run_name}.onnx")
          exported_files.append(f"{run_name}.onnx")
        if hasattr(runner, "export_policy_to_onnx"):
          runner.export_policy_to_onnx(export_dir, "policy.onnx")
          exported_files.append("policy.onnx")
        if exported_files:
          print(
            f"[WARN]: save() unavailable in play mode ({e}). Fallback exported: {', '.join(exported_files)}"
          )
        else:
          raise
    policy = runner.get_inference_policy(device=device)

  if HEADLESS_VIDEO:
    video_out_dir = _find_video_recorder(env)
    video_out_dir = video_out_dir.video_folder if video_out_dir else video_folder
    _run_headless_video(env, policy, cfg.video_length)
    env.close()
    print(f"[INFO] Done. Saved to {video_out_dir}/", flush=True)
    return

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    viewer_cls = (
      StableCameraNativeViewer if cfg.stable_camera else NativeMujocoViewer
    )
    if "Kick" in task_id:
      from src.tasks.kick.rl.native_viewer import KickNativePlayViewer

      viewer_cls = KickNativePlayViewer
    if cfg.print_live_state:
      viewer_cls = wrap_viewer_with_live_state_print(
        viewer_cls, print_hz=cfg.print_live_state_hz
      )
      print(
        f"[INFO] Live state print enabled ({cfg.print_live_state_hz:.1f} Hz): "
        "tau_sum = sum(|actuator torque|)"
      )
    viewer_cls(env, policy).run()
  elif resolved_viewer == "viser":
    import viser

    viewer_cls = ViserPlayViewer
    if "Kick" in task_id:
      from src.tasks.kick.rl.viser_viewer import KickViserPlayViewer

      viewer_cls = KickViserPlayViewer
    if cfg.print_live_state:
      viewer_cls = wrap_viewer_with_live_state_print(
        viewer_cls, print_hz=cfg.print_live_state_hz
      )
      print(
        f"[INFO] Live state print enabled ({cfg.print_live_state_hz:.1f} Hz): "
        "tau_sum = sum(|actuator torque|)"
      )
    viser_server = viser.ViserServer(
      host=cfg.viser_host,
      port=int(cfg.viser_port),
      label="mjlab",
    )
    _print_viser_access_urls(cfg.viser_host, int(cfg.viser_port))
    share_proc = _start_cloudflared_share(int(cfg.viser_port)) if cfg.viser_share else None
    try:
      viewer_cls(env, policy, viser_server=viser_server).run()
    finally:
      if share_proc is not None and share_proc.poll() is None:
        share_proc.terminate()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=PLAY_TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
