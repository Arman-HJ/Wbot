#!/usr/bin/env python3
"""Train TinyMDM motion prior for SMP (offline self-supervised, no RL).

Logs (under ``--out-dir``):
  - ``log.txt``           text metrics
  - ``events.out.tfevents.*``  TensorBoard (loss / lr / time)
  - ``metrics.csv``       iter,loss,lr,sec
  - ``model.pt``          latest checkpoint
  - ``model_{iter}.pt``   periodic snapshots
  - ``diffusion_config.yaml`` / ``env_config.yaml``

Example:
  python scripts/train_smp_prior.py \\
    --cfg src/tasks/smp/configs/tinymdm_adam_sp.yaml \\
    --out-dir logs/smp_prior/engineai_locomotion \\
    --amp-data-path src/assets/amp_motions/pnd_adam_sp/engineai_locomotion
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from src.tasks.smp.prior_dataset import EnvDiscPriorDataset
from src.tasks.smp.tinymdm.tinymdm_model import TinyMDMModel
from rsl_rl.utils.diff_normalizer import DiffNormalizer


def fixseed(seed: int) -> None:
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def parse_datasets(s: str | None) -> dict[str, float] | None:
  if not s:
    return None
  out: dict[str, float] = {}
  for part in s.split(","):
    name, _, w = part.partition(":")
    name = name.strip()
    if not name:
      continue
    out[name] = float(w.strip() or "1.0")
  return out or None


def robot_from_env_task(env_task: str) -> str:
  if "X2" in env_task or "x2" in env_task.lower():
    return "agibot_x2"
  return "adam_sp"


class PriorLogger:
  """Prior-only logger: console + log.txt + CSV + TensorBoard."""

  def __init__(self, out_dir: Path) -> None:
    self.out_dir = out_dir
    self.log_path = out_dir / "log.txt"
    self.csv_path = out_dir / "metrics.csv"
    self._fh = open(self.log_path, "a", encoding="utf-8")
    self._csv = open(self.csv_path, "a", newline="", encoding="utf-8")
    self._csv_writer = csv.writer(self._csv)
    if self.csv_path.stat().st_size == 0:
      self._csv_writer.writerow(
        [
          "iter",
          "loss",
          "lr",
          "sec_per_iter",
          "elapsed_sec",
          "eval_sds_raw",
          "eval_smp_reward",
        ]
      )
      self._csv.flush()

    self.writer = None
    self._num_iters = 0
    try:
      from torch.utils.tensorboard import SummaryWriter

      self.writer = SummaryWriter(log_dir=str(out_dir))
    except Exception as exc:  # noqa: BLE001
      self.info(f"[WARN] TensorBoard unavailable: {exc}")

  def info(self, msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    self._fh.write(line)
    self._fh.flush()

  def log_iter(
    self,
    *,
    it: int,
    loss: float,
    lr: float,
    sec_per_iter: float,
    elapsed_sec: float,
    eval_sds_raw: float | None = None,
    eval_smp_reward: float | None = None,
  ) -> None:
    eta = max(0.0, (self._num_iters - it - 1) * sec_per_iter) if self._num_iters else 0.0
    eval_str = ""
    if eval_sds_raw is not None and eval_smp_reward is not None:
      eval_str = (
        f"  eval_sds={eval_sds_raw:.4f}  eval_smp_r={eval_smp_reward:.4f}"
      )
    self.info(
      f"[prior] iter={it:6d}/{self._num_iters}  loss={loss:.6f}  "
      f"lr={lr:.2e}  {sec_per_iter*1000:.1f} ms/iter  "
      f"elapsed={timedelta(seconds=int(elapsed_sec))}  "
      f"ETA={timedelta(seconds=int(eta))}{eval_str}"
    )
    self._csv_writer.writerow(
      [
        it,
        f"{loss:.8f}",
        f"{lr:.8e}",
        f"{sec_per_iter:.6f}",
        f"{elapsed_sec:.3f}",
        "" if eval_sds_raw is None else f"{eval_sds_raw:.8f}",
        "" if eval_smp_reward is None else f"{eval_smp_reward:.8f}",
      ]
    )
    self._csv.flush()
    if self.writer is not None:
      self.writer.add_scalar("prior/loss", loss, it)
      self.writer.add_scalar("prior/lr", lr, it)
      self.writer.add_scalar("prior/sec_per_iter", sec_per_iter, it)
      if eval_sds_raw is not None:
        self.writer.add_scalar("prior/eval_sds_loss_raw", eval_sds_raw, it)
      if eval_smp_reward is not None:
        self.writer.add_scalar("prior/eval_smp_reward", eval_smp_reward, it)

  def set_num_iters(self, n: int) -> None:
    self._num_iters = int(n)

  def close(self) -> None:
    self._fh.close()
    self._csv.close()
    if self.writer is not None:
      self.writer.flush()
      self.writer.close()


class TrainingDataSdsEvaluator:
  """SDS / smp_reward on fixed training windows (same formula as ``PPOSMP``)."""

  def __init__(
    self,
    *,
    diffusion_steps: list[int],
    sds_loss_scale: float,
    device: str,
  ) -> None:
    self.diffusion_steps = list(diffusion_steps)
    self.sds_loss_scale = float(sds_loss_scale)
    self._sds_normalizer = DiffNormalizer(
      [len(self.diffusion_steps)], device=device, dtype=torch.float32
    )

  @torch.no_grad()
  def evaluate(
    self,
    model: TinyMDMModel,
    windows_hd: torch.Tensor,
    *,
    record: bool = False,
  ) -> dict[str, float]:
    """``windows_hd``: ``(N, H, D)`` raw demo windows from ``AmpSmpPriorDataset``."""
    was_training = model.training
    model.eval()
    norm = model.normalize(windows_hd)
    flat = norm.reshape(windows_hd.shape[0], -1)
    sds_losses = model.ESM_SDS_loss(norm_x_obs=flat, t_lst=self.diffusion_steps)
    if record:
      self._sds_normalizer.record(sds_losses)
      self._sds_normalizer.update()
    sds_norm = self._sds_normalizer.normalize(sds_losses)
    mean_sds_norm = torch.mean(sds_norm, dim=-1)
    smp_r = torch.exp(-mean_sds_norm * self.sds_loss_scale)
    raw_mean = torch.mean(sds_losses, dim=-1)
    if was_training:
      model.train()
    return {
      "eval_sds_raw": float(raw_mean.mean().item()),
      "eval_sds_std": float(raw_mean.std().item()) if raw_mean.numel() > 1 else 0.0,
      "eval_smp_reward": float(smp_r.mean().item()),
    }


def train(args: argparse.Namespace) -> None:
  fixseed(args.seed)
  cfg_path = Path(args.cfg).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  logger = PriorLogger(out_dir)
  t0 = time.time()

  with open(cfg_path, "r") as f:
    config = yaml.safe_load(f)

  env_cfg_src = Path(config["env_config"])
  if not env_cfg_src.is_absolute():
    cand = cfg_path.parent / env_cfg_src
    env_cfg_src = cand if cand.is_file() else Path(env_cfg_src).resolve()

  out_env = out_dir / "env_config.yaml"
  shutil.copy(env_cfg_src, out_env)
  config["env_config"] = str(out_env)
  config["device"] = args.device or config.get("device", "cuda")

  amp_data = Path(args.amp_data_path).expanduser().resolve()
  datasets = parse_datasets(args.amp_datasets)
  sim_dt = float(args.simulation_dt)
  num_steps = int(yaml.safe_load(out_env.read_text())["num_disc_obs_steps"])

  logger.info("=" * 72)
  logger.info("SMP TinyMDM prior pretrain (env disc obs, self-supervised)")
  logger.info(f"time: {datetime.now().isoformat(timespec='seconds')}")
  logger.info(f"out_dir: {out_dir}")
  logger.info(f"cfg: {cfg_path}")
  logger.info(f"amp_data: {amp_data}")
  logger.info(f"amp_datasets: {datasets or '(auto equal-weight npz)'}")
  logger.info(f"device: {config['device']}  seed: {args.seed}")
  logger.info(f"env_task: {args.env_task}")
  logger.info("=" * 72)

  dataset = EnvDiscPriorDataset(
    amp_data_path=amp_data,
    datasets=datasets,
    simulation_dt=sim_dt,
    num_disc_obs_steps=num_steps,
    slow_down_factor=int(args.slow_down_factor),
    device=config["device"],
    env_task=args.env_task,
  )
  try:
    logger.info(dataset.summary())
    feat = dataset.feature_stats()
    logger.info(
      "[env disc] latest-frame stats: "
      f"ang_vel μ={feat['ang_vel_mean']:.3f} σ={feat['ang_vel_std']:.3f} | "
      f"joint_pos μ={feat['joint_pos_mean']:.3f} σ={feat['joint_pos_std']:.3f} | "
      f"joint_vel μ={feat['joint_vel_mean']:.3f} σ={feat['joint_vel_std']:.3f}"
    )
    if dataset.input_channel != 61:
      logger.info(
        f"[WARN] expected input_channel=61 for no-head disc (3+29+29), "
        f"got {dataset.input_channel}"
      )

    robot = robot_from_env_task(args.env_task)
    config["input_dim"] = dataset.input_dim
    config["input_channel"] = dataset.input_channel
    config["num_disc_obs_steps"] = num_steps
    config["control_freq"] = int(round(1.0 / sim_dt))
    config["amp_data_path"] = str(amp_data)
    config["disc_source"] = "env"
    config["env_task"] = args.env_task
    if datasets is not None:
      config["amp_datasets"] = datasets

    out_cfg = out_dir / "diffusion_config.yaml"
    with open(out_cfg, "w") as f:
      yaml.dump(config, f)

    meta = {
      "task": "smp_prior_pretrain",
      "robot": robot,
      "env_task": args.env_task,
      "disc_source": "env",
      "input_dim": dataset.input_dim,
      "input_channel": dataset.input_channel,
      "num_disc_obs_steps": num_steps,
      "num_windows": dataset._num_windows,
      "clip_lengths": dataset.clip_lengths,
      "amp_data_path": str(amp_data),
      "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    batch_size = int(config["batch_size"])
    num_samples_stat = int(config.get("num_samples_stat", 10_000))
    output_iter = int(config.get("output_iter", 2_000))
    log_iter = int(args.log_iter or config.get("log_iter", max(100, output_iter // 10)))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    num_iters = int(args.num_iterations or config["num_iterations"])
    logger.set_num_iters(num_iters)

    samples = dataset.fetch_obs_demo(min(num_samples_stat, dataset._num_windows))
    model = TinyMDMModel(config)
    model.update_normalizer(samples.reshape(-1, num_steps, dataset.input_channel))
    model.to(config["device"])
    optimizer = optim.AdamW(
      filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["lr"])
    )

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
      f"model: TinyMDM/{config.get('arch_name')}  params={n_params / 1e6:.2f}M  "
      f"batch={batch_size}  T={config.get('T')}  lr={config['lr']}  "
      f"iters={num_iters}  log_every={log_iter}  ckpt_every={output_iter}"
    )
    if logger.writer is not None:
      logger.writer.add_text("prior/summary", dataset.summary(), 0)
      logger.writer.add_scalar("prior/num_params_M", n_params / 1e6, 0)
      logger.writer.add_scalar("prior/num_windows", dataset._num_windows, 0)

    eval_batch_size = int(config.get("eval_sds_batch_size", 4096))
    eval_seed = int(config.get("eval_sds_seed", 42))
    diffusion_steps = list(config.get("diffusion_steps", [22, 15, 8]))
    sds_loss_scale = float(config.get("sds_loss_scale", 6.0))
    eval_indices = dataset.make_eval_indices(eval_batch_size, seed=eval_seed)
    eval_windows = dataset.gather_windows_hd(eval_indices)
    sds_eval = TrainingDataSdsEvaluator(
      diffusion_steps=diffusion_steps,
      sds_loss_scale=sds_loss_scale,
      device=config["device"],
    )
    init_eval = sds_eval.evaluate(model, eval_windows, record=True)
    logger.info(
      "[eval] init (untrained) on "
      f"{eval_windows.shape[0]} env-disc windows: "
      f"sds_raw={init_eval['eval_sds_raw']:.4f}  "
      f"smp_r={init_eval['eval_smp_reward']:.4f}  "
      f"(↑ smp_r = prior matches env disc; normalizer fixed after this)"
    )
    if logger.writer is not None:
      logger.writer.add_scalar("prior/eval_sds_loss_raw", init_eval["eval_sds_raw"], 0)
      logger.writer.add_scalar("prior/eval_smp_reward", init_eval["eval_smp_reward"], 0)

    out_model = out_dir / "model.pt"
    loss_sum = 0.0
    loss_count = 0
    t_loop = time.time()

    for it in range(num_iters):
      batch = dataset.fetch_obs_demo(batch_size).to(config["device"])
      batch = model.normalize(batch.reshape(batch_size, num_steps, -1)).reshape(
        batch_size, -1
      )
      loss = model(batch)
      loss_val = float(loss.item())
      loss_sum += loss_val
      loss_count += 1

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
      optimizer.step()
      if model.model_ema:
        model.ema_dmodel.update()

      do_log = (it + 1) % log_iter == 0 or it == 0 or it == num_iters - 1
      do_ckpt = ((it + 1) % output_iter == 0) or (it == num_iters - 1)

      if do_log:
        avg = loss_sum / max(1, loss_count)
        elapsed = time.time() - t0
        sec_per = (time.time() - t_loop) / max(1, loss_count)
        lr = float(optimizer.param_groups[0]["lr"])
        eval_metrics = sds_eval.evaluate(model, eval_windows, record=False)
        logger.log_iter(
          it=it,
          loss=avg,
          lr=lr,
          sec_per_iter=sec_per,
          elapsed_sec=elapsed,
          eval_sds_raw=eval_metrics["eval_sds_raw"],
          eval_smp_reward=eval_metrics["eval_smp_reward"],
        )
        loss_sum = 0.0
        loss_count = 0
        t_loop = time.time()

      if do_ckpt:
        torch.save(model.state_dict(), out_model)
        snap = out_dir / f"model_{it}.pt"
        torch.save(model.state_dict(), snap)
        logger.info(f"[prior] checkpoint → {out_model.name}  snapshot → {snap.name}")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_sec"] = time.time() - t0
    meta["final_model"] = str(out_model)
    final_eval = sds_eval.evaluate(model, eval_windows, record=False)
    meta["final_eval_sds_raw"] = final_eval["eval_sds_raw"]
    meta["final_eval_smp_reward"] = final_eval["eval_smp_reward"]
    logger.info(
      f"[eval] final on env disc: sds_raw={final_eval['eval_sds_raw']:.4f}  "
      f"smp_r={final_eval['eval_smp_reward']:.4f}"
    )
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[prior] done. model={out_model}  config={out_cfg}")
    logger.info(f"[prior] tensorboard: tensorboard --logdir {out_dir}")
  finally:
    dataset.close()
    logger.close()


def main() -> None:
  p = argparse.ArgumentParser(description="Train SMP TinyMDM prior (self-supervised)")
  p.add_argument(
    "--cfg",
    type=str,
    default="src/tasks/smp/configs/tinymdm_adam_sp.yaml",
  )
  p.add_argument("--out-dir", type=str, required=True)
  p.add_argument(
    "--amp-data-path",
    type=str,
    default="src/assets/amp_motions/pnd_adam_sp/engineai_locomotion",
  )
  p.add_argument("--amp-datasets", type=str, default="")
  p.add_argument("--simulation-dt", type=float, default=0.02)
  p.add_argument("--slow-down-factor", type=int, default=1)
  p.add_argument(
    "--env-task",
    type=str,
    default="AdamSP-SMP-Flat",
    help="MuJoCo task for env disc replay (e.g. AgiBot-X2-SMP-Flat)",
  )
  p.add_argument("--device", type=str, default="")
  p.add_argument("--num-iterations", type=int, default=0)
  p.add_argument("--log-iter", type=int, default=0, help="Console/TB log interval")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  if not args.num_iterations:
    args.num_iterations = None
  if not args.log_iter:
    args.log_iter = None
  train(args)


if __name__ == "__main__":
  main()
