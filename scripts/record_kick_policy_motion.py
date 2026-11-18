#!/usr/bin/env python3
"""Roll a kick checkpoint once and save the realized motion as training npz.

The clip is the policy rollout (not the kinematic reference). After the kick,
the final pose is held for an extra second so the trailing stand is longer.

Example:
  python scripts/record_kick_policy_motion.py \\
    --checkpoint-file logs/rsl_rl/wbot_20dof_kick/.../model_1500.pt \\
    --motion-file src/assets/motions/wbot_20dof/gvhmr_kick/kick4_wbot_20dof.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def _finite_diff(pos: np.ndarray, dt: float) -> np.ndarray:
  vel = np.zeros_like(pos)
  vel[1:] = (pos[1:] - pos[:-1]) / dt
  vel[0] = vel[1]
  return vel


def _ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """World angular velocity from wxyz quaternions. quat: (T, B, 4)."""
  t_len, n_body, _ = quat_wxyz.shape
  ang = np.zeros((t_len, n_body, 3), dtype=np.float64)
  for b in range(n_body):
    rot = R.from_quat(quat_wxyz[:, b][:, [1, 2, 3, 0]])
    if t_len < 2:
      continue
    ang[1:, b] = (rot[1:] * rot[:-1].inv()).as_rotvec() / dt
  ang[0] = ang[1]
  return ang


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(
      "src/assets/motions/wbot_20dof/gvhmr_kick/kick4_wbot_20dof_inferred.npz"
    ),
  )
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--stand-extension-s",
    type=float,
    default=1.0,
    help="Hold the final pose this many extra seconds.",
  )
  parser.add_argument(
    "--start-frame",
    type=int,
    default=0,
    help="Drop rollout frames before this index (0-based) before saving.",
  )
  args = parser.parse_args()

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
  # Nominal robot: drop pushes and the play-time startup/reset randomization
  # so the saved clip is not one random friction / CoM / PD draw.
  env_cfg.events = {}
  motion_cmd = env_cfg.commands["motion"]
  motion_cmd.motion_file = str(args.motion_file.resolve())
  motion_cmd.ball_pos_noise = (0.0, 0.0, 0.0)
  motion_cmd.joint_position_range = (0.0, 0.0)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  clip_actions = getattr(agent_cfg, "clip_actions", None)
  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  runner = MjlabOnPolicyRunner(
    env,
    asdict(agent_cfg),
    log_dir=str(args.checkpoint_file.parent),
    device=device,
  )
  runner.load(str(args.checkpoint_file), map_location=device)
  policy = runner.get_inference_policy(device=device)

  base = env.unwrapped
  cmd = base.command_manager.get_term("motion")
  assert isinstance(cmd, KickMotionCommand)
  robot = cmd.robot
  dt = float(base.step_dt)
  fps = 1.0 / dt
  ref = np.load(args.motion_file)
  joint_names = [str(n) for n in ref["joint_names"]]
  body_names = [str(n) for n in ref["body_names"]]
  joint_index = [robot.joint_names.index(n) for n in joint_names]
  body_index = [robot.body_names.index(n) for n in body_names]
  origin = base.scene.env_origins[0].detach()

  def snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jp = robot.data.joint_pos[0, joint_index].detach().cpu().numpy()
    pos = robot.data.body_link_pos_w[0, body_index].detach() - origin
    quat = robot.data.body_link_quat_w[0, body_index].detach()
    return (
      jp.astype(np.float64),
      pos.cpu().numpy().astype(np.float64),
      quat.cpu().numpy().astype(np.float64),
    )

  env.reset()
  cmd.start_play_clip()
  obs = env.get_observations()
  joint_pos = []
  body_pos = []
  body_quat = []
  # Frame 0 is the reset pose; later frames are after each policy step.
  jp0, bp0, bq0 = snapshot()
  joint_pos.append(jp0)
  body_pos.append(bp0)
  body_quat.append(bq0)

  t_ref = int(cmd.motion.time_step_total)
  for _ in range(t_ref + 5):
    with torch.inference_mode():
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
    jp, bp, bq = snapshot()
    joint_pos.append(jp)
    body_pos.append(bp)
    body_quat.append(bq)
    if not cmd._clip_playing:
      break

  jp = np.stack(joint_pos)
  bp = np.stack(body_pos)
  bq = np.stack(body_quat)
  n_roll = jp.shape[0]
  start = int(args.start_frame)
  if start < 0 or start >= n_roll:
    raise SystemExit(f"--start-frame {start} out of range for rollout length {n_roll}")
  if start > 0:
    jp = jp[start:]
    bp = bp[start:]
    bq = bq[start:]
  n_kept = jp.shape[0]
  extra = int(round(args.stand_extension_s / dt))
  if extra > 0:
    jp = np.concatenate([jp, np.repeat(jp[-1:], extra, axis=0)], axis=0)
    bp = np.concatenate([bp, np.repeat(bp[-1:], extra, axis=0)], axis=0)
    bq = np.concatenate([bq, np.repeat(bq[-1:], extra, axis=0)], axis=0)

  jv = _finite_diff(jp, dt)
  lv = _finite_diff(bp, dt)
  av = _ang_vel(bq, dt)
  if extra > 0:
    jv[-extra:] = 0.0
    lv[-extra:] = 0.0
    av[-extra:] = 0.0

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    fps=np.array([fps], dtype=np.float32),
    joint_names=np.array(joint_names),
    body_names=np.array(body_names),
    joint_pos=jp.astype(np.float32),
    joint_vel=jv.astype(np.float32),
    body_pos_w=bp.astype(np.float32),
    body_quat_w=bq.astype(np.float32),
    body_lin_vel_w=lv.astype(np.float32),
    body_ang_vel_w=av.astype(np.float32),
    source=np.array(f"policy_rollout:{args.checkpoint_file}"),
    reference_motion=np.array(str(args.motion_file)),
    start_frame=np.array([start], dtype=np.int32),
  )
  dur = jp.shape[0] / fps
  print(
    f"[record] rollout frames={n_roll}  drop_before={start}  kept={n_kept}"
    f"  +stand {extra} ({args.stand_extension_s:.1f}s)"
    f"  total={jp.shape[0]}  fps={fps:.1f}  dur={dur:.2f}s"
  )
  print(f"[record] wrote {args.output.resolve()}")
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
