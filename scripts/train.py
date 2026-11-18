"""Script to train RL agent with RSL-RL."""

import logging
import os
import re
import shutil
import sys
import copy
from dataclasses import asdict, dataclass, field, fields, make_dataclass
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Literal, cast

import numpy as np
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


def _configure_mujoco_gl_backend() -> None:
  """Configure MuJoCo OpenGL backend before importing mjlab/mujoco modules."""
  if "MUJOCO_GL" not in os.environ:
    # Headless Linux default: EGL. Users can still override via env var.
    os.environ["MUJOCO_GL"] = "egl"
  if os.environ.get("MUJOCO_GL") == "egl" and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  # With both glvnd vendors loaded, EGL device 0 is Mesa's (broken) DRI device on the
  # NVIDIA card and device 1 is llvmpipe (CPU, ~1000x slower). Load only the NVIDIA
  # vendor so device ids map to CUDA GPUs; fall back to Mesa llvmpipe without it.
  if (
    os.environ.get("MUJOCO_GL") == "egl"
    and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ
  ):
    nvidia_vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if nvidia_vendor.is_file() and os.access("/dev/nvidiactl", os.R_OK | os.W_OK):
      os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(nvidia_vendor)
    elif mesa_vendor.is_file():
      os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_vendor)
      os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "1")


_configure_mujoco_gl_backend()

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from src.env_scale_print import (
  build_deployment_alignment_snapshot,
  print_env_input_output_scales,
  print_robot_init_and_defaults,
)
from mjlab.utils.wrappers import VideoRecorder


_BOOTSTRAP_LOG_HANDLE = None


class _Tee:
  """Mirror writes to multiple streams (console + train.log)."""

  def __init__(self, *streams):
    self._streams = streams

  def write(self, data: str) -> None:
    if not data:
      return
    for stream in self._streams:
      stream.write(data)

  def flush(self) -> None:
    for stream in self._streams:
      stream.flush()

  def isatty(self) -> bool:
    return self._streams[0].isatty()

  def fileno(self) -> int:
    return self._streams[0].fileno()

  @property
  def encoding(self) -> str:
    for stream in self._streams:
      enc = getattr(stream, "encoding", None)
      if enc:
        return enc
    return "utf-8"


def _bootstrap_console_log_if_needed() -> None:
  """Capture early stdout/stderr when launched via nohup (before run log dir exists)."""
  global _BOOTSTRAP_LOG_HANDLE
  if sys.stdout.isatty() and sys.stderr.isatty():
    return
  bootstrap_dir = Path("logs/rsl_rl/_console")
  bootstrap_dir.mkdir(parents=True, exist_ok=True)
  log_path = bootstrap_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{os.getpid()}.log"
  _BOOTSTRAP_LOG_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
  sys.stdout = _Tee(sys.__stdout__, _BOOTSTRAP_LOG_HANDLE)
  sys.stderr = _Tee(sys.__stderr__, _BOOTSTRAP_LOG_HANDLE)
  print(f"[INFO] Bootstrap console log: {log_path.resolve()}", flush=True)


def _setup_train_console_log(log_path: Path) -> None:
  global _BOOTSTRAP_LOG_HANDLE
  log_path.parent.mkdir(parents=True, exist_ok=True)
  log_file = open(log_path, "a", encoding="utf-8", buffering=1)
  if _BOOTSTRAP_LOG_HANDLE is not None:
    _BOOTSTRAP_LOG_HANDLE.close()
    _BOOTSTRAP_LOG_HANDLE = None
  sys.stdout = _Tee(sys.__stdout__, log_file)
  sys.stderr = _Tee(sys.__stderr__, log_file)
  print(f"[INFO] Console log: {log_path.resolve()}", flush=True)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  checkpoint_file: str | None = None
  """Resume from this ``.pt`` path (overrides ``agent.resume`` / ``load_run``)."""
  video: bool = True
  video_length: int = 200
  video_interval: int = 2000
  video_width: int = 1280
  video_height: int = 720
  enable_nan_guard: bool = True
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _train_config_cls_for_agent(agent_type: type) -> type:
  """Build a tyro config class with a concrete ``agent`` type (avoids Union warnings)."""
  return make_dataclass(
    "TrainConfig",
    [
      ("env", ManagerBasedRlEnvCfg, field()),
      ("agent", agent_type, field()),
      ("motion_file", str | None, field(default=None)),
      ("checkpoint_file", str | None, field(default=None)),
      ("video", bool, field(default=False)),
      ("video_length", int, field(default=200)),
      ("video_interval", int, field(default=2000)),
      ("video_width", int, field(default=1280)),
      ("video_height", int, field(default=720)),
      ("enable_nan_guard", bool, field(default=True)),
      ("torchrunx_log_dir", str | None, field(default=None)),
      (
        "gpu_ids",
        list[int] | Literal["all"] | None,
        field(default_factory=lambda: [0]),
      ),
    ],
    frozen=True,
  )


def _tyro_default(train_config_cls: type, default_cfg: TrainConfig):
  """Build tyro default without ``asdict`` (nested dataclasses must stay typed)."""
  return train_config_cls(
    **{f.name: getattr(default_cfg, f.name) for f in fields(TrainConfig)}
  )


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


def _print_joint_pd_values(cfg: TrainConfig, env: ManagerBasedRlEnv) -> None:
  robot = env.scene["robot"]
  robot_cfg = cfg.env.scene.entities.get("robot")
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


def _apply_training_video_viewer(
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  video_width: int,
  video_height: int,
) -> None:
  """Fixed overview camera for train videos: no body tracking, many envs visible."""
  from mjlab.viewer import ViewerConfig

  num_envs = int(env_cfg.scene.num_envs)
  viewer = env_cfg.viewer
  viewer.width = int(video_width)
  viewer.height = int(video_height)
  viewer.origin_type = ViewerConfig.OriginType.WORLD
  viewer.body_name = None
  viewer.fovy = 40.0
  # Wide overview: render nearest neighbors around env_idx. Cap extra envs hard —
  # Mesa EGL is CPU soft-render; 127 extras at 1280x720 blocks training for minutes.
  viewer.max_extra_envs = min(15, max(0, num_envs - 1))
  spacing = float(env_cfg.scene.env_spacing)
  viewer.distance = max(8.0, spacing * 4.0)
  viewer.elevation = -25.0
  viewer.azimuth = 135.0
  print(
    "[INFO] Training video camera: WORLD overview "
    f"({viewer.width}x{viewer.height}, max_extra_envs={viewer.max_extra_envs}, "
    f"distance≈{viewer.distance}).",
    flush=True,
  )


def _configure_training_video_camera(env: ManagerBasedRlEnv) -> None:
  """Point the offscreen camera at the env grid using real ``env_origins``."""
  renderer = env._offline_renderer
  if renderer is None:
    return

  origins = env.scene.env_origins[: env.num_envs].detach().cpu().numpy()
  spacing = float(env.cfg.scene.env_spacing)
  centroid = origins.mean(axis=0)
  dist2 = np.sum((origins[:, :2] - centroid[:2]) ** 2, axis=1)
  env_idx = int(np.argmin(dist2))

  lookat = (
    float(origins[env_idx, 0]),
    float(origins[env_idx, 1]),
    0.9,
  )
  distance = max(10.0, spacing * 6.0)

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

  print(
    "[INFO] Training video lookat="
    f"({lookat[0]:.1f}, {lookat[1]:.1f}, {lookat[2]:.1f}), "
    f"env_idx={env_idx}, distance={distance:.1f}",
    flush=True,
  )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # NVIDIA EGL: bind device to this rank's CUDA GPU.
    # Mesa llvmpipe fallback: keep preconfigured device id 1; device 0 is Mesa's
    # broken DRI device on the NVIDIA card and ImportError's.
    mesa_vendor = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "").lower()
    if "mesa" not in mesa_vendor:
      os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if cfg.motion_file:
      motion_path = Path(cfg.motion_file).expanduser().resolve()
      if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
      motion_cmd.motion_file = str(motion_path)
      print(f"[INFO] Using motion file from --motion-file: {motion_cmd.motion_file}")
    else:
      # Allow task/env defaults to provide motion file (e.g. velocity_v2 X2 cfg).
      if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
        print(f"[INFO] Using motion file from env cfg: {motion_cmd.motion_file}")
      else:
        raise ValueError(
          "For tracking tasks, provide --motion-file or set a valid "
          "env.commands.motion.motion_file in task env cfg."
        )

    # Early sanity checks to avoid opaque CUDA device-side assert later.
    motion_npz = np.load(motion_cmd.motion_file)
    required_keys = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
    missing = required_keys.difference(set(motion_npz.files))
    if missing:
      raise ValueError(
        f"Invalid motion file: missing keys {sorted(missing)} in {motion_cmd.motion_file}"
      )
    body_dim = int(motion_npz["body_pos_w"].shape[1])
    expected_bodies = len(motion_cmd.body_names)
    if expected_bodies > 0 and body_dim < expected_bodies:
      raise ValueError(
        "Motion body dimension mismatch: "
        f"body_pos_w has {body_dim} bodies, but env expects at least {expected_bodies} "
        f"from commands.motion.body_names. Motion file: {motion_cmd.motion_file}"
      )
    # Snapshot motion into this run's log dir (self-contained experiment).
    motion_cmd.motion_file = str(_copy_motion_file_to_log(motion_cmd.motion_file, log_dir))

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    # Default /tmp/mjlab/nan_dumps is often owned by another user on shared boxes;
    # dump into this run's log dir so PermissionError cannot kill training.
    nan_dir = Path(log_dir) / "nan_dumps"
    nan_dir.mkdir(parents=True, exist_ok=True)
    cfg.env.sim.nan_guard.output_dir = str(nan_dir)
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  if cfg.video:
    _apply_training_video_viewer(
      cfg.env,
      video_width=cfg.video_width,
      video_height=cfg.video_height,
    )

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if cfg.video:
    _configure_training_video_camera(env)
  _print_joint_pd_values(cfg, env)
  deployment_alignment: dict | None = None
  if rank == 0:
    print_robot_init_and_defaults(cfg.env, env)
    print_env_input_output_scales(cfg.env, cfg.agent)
    deployment_alignment = build_deployment_alignment_snapshot(cfg.env, env, task_id=task_id)
    if is_tracking_task:
      motion_cmd = cfg.env.commands["motion"]
      deployment_alignment.setdefault("meta", {})["motion_file"] = getattr(
        motion_cmd, "motion_file", None
      )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not resume_path.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  elif cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      # Skip step 0: Mesa soft-render of the first clip blocks learning for minutes.
      step_trigger=lambda step: step > 0 and step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  agent_cfg = asdict(cfg.agent)
  clip_actions = cfg.agent.clip_actions
  max_iterations = cfg.agent.max_iterations

  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  env_cfg = asdict(cfg.env)
  # Keep a YAML-serializable snapshot before runner construction mutates algorithm cfg
  # (e.g. symmetry extension injects non-serializable env objects into agent_cfg).
  agent_cfg_dump = copy.deepcopy(agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  runner.add_git_repo_to_log(__file__)
  learn_iterations = max_iterations
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))
    learn_iterations = max_iterations - runner.current_learning_iteration
    if learn_iterations <= 0:
      raise ValueError(
        f"max_iterations={max_iterations} <= checkpoint iter "
        f"{runner.current_learning_iteration}; nothing to train."
      )
    print(
      f"[INFO]: Resume from iter {runner.current_learning_iteration}, "
      f"training {learn_iterations} more steps (target {max_iterations})."
    )

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dump)
    if deployment_alignment is not None:
      dump_yaml(log_dir / "params" / "deployment_alignment.yaml", deployment_alignment)

  runner.learn(
    num_learning_iterations=learn_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ.setdefault("MUJOCO_GL", "egl")

  if num_gpus <= 1:
    # Mirror stdout/stderr into the run directory for nohup / background jobs.
    _setup_train_console_log(log_dir / "train.log")
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  _bootstrap_console_log_if_needed()
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

  default_cfg = TrainConfig.from_task(chosen_task)
  train_config_cls = _train_config_cls_for_agent(type(default_cfg.agent))
  args = tyro.cli(
    train_config_cls,
    args=remaining_args,
    default=_tyro_default(train_config_cls, default_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
