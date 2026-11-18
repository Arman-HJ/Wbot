#!/usr/bin/env python3
"""Headless kick eval: EE / ankle tracking vs motion + ball displacement.

Usage:
  python scripts/eval_kick_tracking.py \\
    --checkpoint-file logs/rsl_rl/wbot_20dof_kick/.../model_30000.pt \\
    --motion-file src/assets/motions/wbot_20dof/gvhmr_kick/kick3_....npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, default=None)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--steps", type=int, default=0, help="0 = full motion once")
  args = parser.parse_args()

  # Register tasks.
  from dataclasses import asdict

  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.kick.mdp.commands import KickMotionCommand

  configure_torch_backends()
  device = args.device if torch.cuda.is_available() else "cpu"
  task_id = "Wbot-20Dof-Kick"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  # Drop pushes.
  for name in list(env_cfg.events):
    if name.startswith("push"):
      del env_cfg.events[name]
  if args.motion_file is not None:
    env_cfg.commands["motion"].motion_file = str(args.motion_file.resolve())

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  clip_actions = getattr(agent_cfg, "clip_actions", None)
  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  log_dir = args.checkpoint_file.parent
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
  runner.load(str(args.checkpoint_file), map_location=device)
  policy = runner.get_inference_policy(device=device)

  base = env.unwrapped
  cmd = base.command_manager.get_term("motion")
  assert isinstance(cmd, KickMotionCommand)

  # Body indices in command.
  body_names = list(cmd.cfg.body_names)
  ee_names = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
  ]
  ee_idx = [body_names.index(n) for n in ee_names if n in body_names]
  right_ankle = body_names.index("right_ankle_roll_link")
  T = int(cmd.motion.time_step_total)
  n_steps = args.steps or (T + 20)
  print(f"[eval] motion T={T}, running {n_steps} policy steps, device={device}")

  obs = env.get_observations()
  # Start Play Clip (freeze align + advance).
  ok = cmd.start_play_clip()
  print(f"[eval] start_play_clip -> {ok}")

  ball = base.scene["ball"]
  ball0 = ball.data.root_link_pos_w[0].detach().cpu().numpy().copy()

  err_ee = []
  err_right = []
  err_body = []
  times = []
  frames = []
  ball_xy = []
  right_robot = []
  right_ref = []

  for step in range(n_steps):
    with torch.inference_mode():
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)

    # Relative ref vs robot (same space as rewards).
    ref = cmd.body_pos_relative_w[0]  # (B, 3)
    rob = cmd.robot_body_pos_w[0]
    e = (ref - rob).norm(dim=-1)  # (B,)
    err_body.append(float(e.mean().cpu()))
    err_ee.append(float(e[ee_idx].mean().cpu()))
    err_right.append(float(e[right_ankle].cpu()))
    times.append(float(base.episode_length_buf[0].cpu()) * base.step_dt)
    frames.append(int(cmd.time_steps[0].cpu()))
    bp = ball.data.root_link_pos_w[0].detach().cpu().numpy()
    ball_xy.append(bp[:2].copy())
    right_robot.append(rob[right_ankle].detach().cpu().numpy().copy())
    right_ref.append(ref[right_ankle].detach().cpu().numpy().copy())

    if bool(dones[0].cpu()) and step > 5:
      print(f"[eval] episode done at step {step}")
      break
    if hasattr(cmd, "_clip_playing") and not cmd._clip_playing and step > 10:
      # Clip finished and returned to hold — keep a few more steps for ball.
      if step > T + 5:
        break

  err_ee = np.asarray(err_ee)
  err_right = np.asarray(err_right)
  err_body = np.asarray(err_body)
  ball_xy = np.asarray(ball_xy)
  right_robot = np.asarray(right_robot)
  right_ref = np.asarray(right_ref)
  frames = np.asarray(frames)

  ball_disp = float(np.linalg.norm(ball_xy[-1] - ball_xy[0]))
  ball_max = float(np.linalg.norm(ball_xy - ball_xy[0], axis=-1).max())
  # Contact proxy: min distance robot right ankle ↔ ball during clip.
  ball_pos = []
  # recompute from logged? we only have ball xy. Use final displacement.

  # Peak kick phase: frames where right foot height or forward progress max.
  # Report mean/p50/p90/max of right ankle error during advancing frames.
  advancing = frames > 0
  if advancing.any():
    ra = err_right[advancing]
    ee = err_ee[advancing]
    bd = err_body[advancing]
  else:
    ra, ee, bd = err_right, err_ee, err_body

  def stats(a: np.ndarray) -> str:
    return (
      f"mean={a.mean():.4f}  p50={np.median(a):.4f}  "
      f"p90={np.percentile(a, 90):.4f}  max={a.max():.4f}"
    )

  # Where right ankle error peaks.
  i_peak = int(np.argmax(err_right))
  print("\n=== Kick tracking eval ===")
  print(f"checkpoint: {args.checkpoint_file}")
  print(f"steps logged: {len(err_right)}  frame range: {frames.min()}→{frames.max()}")
  print(f"body pos err (m):   {stats(bd)}")
  print(f"EE  pos err (m):    {stats(ee)}")
  print(f"right ankle err(m): {stats(ra)}")
  print(
    f"peak right ankle @step={i_peak} frame={frames[i_peak]}: "
    f"err={err_right[i_peak]:.4f} m\n"
    f"  robot={right_robot[i_peak]}  ref={right_ref[i_peak]}  "
    f"Δ={right_robot[i_peak] - right_ref[i_peak]}"
  )
  # Error at mid / impact-ish (frame ~0.4T and 0.6T).
  for frac in (0.3, 0.5, 0.7):
    target = int(frac * (T - 1))
    idxs = np.where(frames == target)[0]
    if len(idxs):
      i = int(idxs[0])
      print(
        f"frame {target} (~{frac:.0%}): right_err={err_right[i]:.4f}  "
        f"Δxyz={right_robot[i] - right_ref[i]}"
      )
  print(f"ball start xy: {ball0[:2]}")
  print(f"ball end   xy: {ball_xy[-1]}")
  print(f"ball xy disp final={ball_disp:.4f} m  max={ball_max:.4f} m")

  # Early vs late tracking.
  mid = max(1, len(ra) // 2)
  print(f"right ankle err first half mean={ra[:mid].mean():.4f}  second={ra[mid:].mean():.4f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
