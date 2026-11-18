"""Download all files from a Weights & Biases run into local log directories."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
LOG_DIR_RE = re.compile(r"(?:^|/)logs/rsl_rl/([^/]+)/")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif", ".avi", ".mov", ".mkv"}

# experiment_name -> local log dir + W&B project (project 可与 experiment 不同)
PRESETS: dict[str, dict[str, str | Path]] = {
  "x2_velocity_v3": {
    "log_path": Path("logs/rsl_rl/x2_velocity_v3"),
    "wandb_project": "X2",
  },
  "x2_amp": {
    "log_path": Path("logs/rsl_rl/x2_amp"),
    "wandb_project": "X2",
  },
  "x2_amp_v2": {
    "log_path": Path("logs/rsl_rl/x2_amp_v2"),
    "wandb_project": "x2_amp_v2",
  },
  "adam_sp_velocity_v3": {
    "log_path": Path("logs/rsl_rl/adam_sp_velocity_v3"),
    "wandb_project": "adam_sp_velocity_v3",
  },
}


def _import_wandb():
  try:
    import wandb
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "需要安装 wandb，例如: pip install wandb"
    ) from exc
  return wandb


def _default_entity(wandb) -> str:
  for key in ("WANDB_ENTITY", "WANDB_USERNAME"):
    value = os.environ.get(key)
    if value:
      return value
  return wandb.Api().default_entity


def _find_runs_by_name(api, entity: str, project: str, run_name: str):
  path = f"{entity}/{project}"
  try:
    return [run for run in api.runs(path) if run.name == run_name]
  except Exception as exc:
    if "Could not find project" in str(exc):
      return []
    raise


def _candidate_projects(experiment: str) -> list[str]:
  preset = PRESETS.get(experiment, {})
  projects: list[str] = []
  wandb_project = preset.get("wandb_project")
  if wandb_project:
    projects.append(str(wandb_project))
  if experiment not in projects:
    projects.append(experiment)
  return list(dict.fromkeys(projects))


def _infer_experiment_from_run(run) -> str | None:
  config = run.config or {}
  train_cfg = config.get("train_cfg") or {}
  experiment = train_cfg.get("experiment_name")
  if isinstance(experiment, str) and experiment in PRESETS:
    return experiment

  log_dir = config.get("log_dir")
  if isinstance(log_dir, str):
    match = LOG_DIR_RE.search(log_dir.replace("\\", "/"))
    if match and match.group(1) in PRESETS:
      return match.group(1)
  return None


def _resolve_output_dir(run, resolved_experiment: str | None) -> Path:
  config = run.config or {}
  log_dir = config.get("log_dir")
  if isinstance(log_dir, str) and log_dir.strip():
    return Path(log_dir.replace("\\", "/"))

  if resolved_experiment is not None:
    return PRESETS[resolved_experiment]["log_path"] / run.name  # type: ignore[index, operator]
  return Path("logs/rsl_rl/wandb_checkpoints") / run.name


def _resolve_run(
  api,
  entity: str,
  experiment: str | None,
  run_name: str | None,
  wandb_run_path: str | None,
):
  if wandb_run_path is not None:
    run = api.run(wandb_run_path)
    return run, None

  if run_name is None:
    raise ValueError("请指定 run 名（如 2026-06-14_11-48-48）或 --wandb-run entity/project/run_id")

  if experiment is not None:
    if experiment not in PRESETS:
      raise ValueError(
        f"未知实验 {experiment!r}，可选: {', '.join(sorted(PRESETS))}"
      )
    matches: list[object] = []
    tried_projects: list[str] = []
    for project in _candidate_projects(experiment):
      tried_projects.append(project)
      matches.extend(_find_runs_by_name(api, entity, project, run_name))
    if not matches:
      raise ValueError(
        f"在 W&B 项目 {entity}/{{{', '.join(tried_projects)}}} 中均未找到 run: {run_name}"
      )
    if len(matches) > 1:
      matches.sort(key=lambda r: r.created_at, reverse=True)  # type: ignore[attr-defined]
      print(
        f"[WARN] 存在多个同名 run，使用最新: {matches[0].id}"  # type: ignore[index]
      )
    return matches[0], experiment

  hits: list[tuple[str, object]] = []
  for project in api.projects(entity):
    for run in _find_runs_by_name(api, entity, project.name, run_name):
      experiment_key = _infer_experiment_from_run(run) or project.name
      hits.append((experiment_key, run))

  if not hits:
    raise ValueError(
      f"账号 {entity} 下所有 W&B 项目均未找到 run: {run_name}\n"
      "请显式指定实验，例如: download_wandb_weights.py pnd_adam_pro_velocity_v3_legs 2026-06-14_11-48-48"
    )
  if len(hits) > 1:
    detail = ", ".join(f"{key} ({run.id})" for key, run in hits)  # type: ignore[attr-defined]
    raise ValueError(
      f"run {run_name} 存在于多个实验: {detail}\n"
      "请显式指定实验名。"
    )
  return hits[0][1], hits[0][0]


def _is_video_file(filename: str) -> bool:
  normalized = filename.replace("\\", "/").lstrip("./")
  if normalized == "videos" or normalized.startswith("videos/"):
    return True
  return Path(normalized).suffix.lower() in VIDEO_EXTENSIONS


def _download_file(wandb_run, filename: str, output_dir: Path) -> Path:
  target = output_dir / filename
  target.parent.mkdir(parents=True, exist_ok=True)
  if target.exists():
    print(f"[SKIP] 已存在: {target}")
    return target
  print(f"[DOWN] {filename} -> {target}")
  wandb_run.file(filename).download(str(output_dir), replace=True)
  return target


def download_run(
  experiment: str | None,
  run_name: str | None,
  wandb_run_path: str | None,
  entity: str | None,
  output_dir: Path | None,
) -> Path:
  wandb = _import_wandb()
  api = wandb.Api()
  resolved_entity = entity or _default_entity(wandb)
  run, resolved_experiment = _resolve_run(
    api, resolved_entity, experiment, run_name, wandb_run_path
  )

  if output_dir is None:
    output_dir = _resolve_output_dir(run, resolved_experiment)
  output_dir = output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  files = list(run.files())
  if not files:
    raise ValueError(f"该 run 在 W&B 上没有任何文件: {run.name} ({run.id})")

  to_download = [file for file in files if not _is_video_file(file.name)]
  skipped = len(files) - len(to_download)
  if not to_download:
    raise ValueError(
      f"该 run 除 video 外无其他文件: {run.name} ({run.id})"
    )

  print("==========================================")
  print("从 W&B 下载 run 文件夹（跳过 video）")
  print("==========================================")
  print(f"Entity : {resolved_entity}")
  print(f"Project: {run.project}")
  print(f"Run    : {run.name} ({run.id})")
  if resolved_experiment:
    print(f"实验   : {resolved_experiment}")
  print(f"输出   : {output_dir}")
  print(f"文件   : {len(to_download)} 个（跳过 video: {skipped} 个）")
  print("==========================================")

  for file in to_download:
    _download_file(run, file.name, output_dir)

  print(f"完成: {output_dir}")
  return output_dir


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="从 Weights & Biases 下载 run 文件夹到本地 logs 目录（跳过 videos/ 及视频文件）。",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
  python scripts/download_wandb_weights.py 2026-06-14_11-48-48
  python scripts/download_wandb_weights.py pnd_adam_pro_velocity_v3_legs 2026-06-14_11-48-48
  python scripts/download_wandb_weights.py --wandb-run entity/pnd_adam_pro_velocity_v3/run_id

需先登录 W&B: wandb login
默认 entity 读取 WANDB_ENTITY / WANDB_USERNAME，否则使用 wandb 默认账号。
""",
  )
  parser.add_argument(
    "experiment",
    nargs="?",
    help="实验名（如 pnd_adam_pro_velocity_v3_legs）；省略时在预设项目中自动查找 run",
  )
  parser.add_argument(
    "run_name",
    nargs="?",
    help="run 目录名 / W&B run name（如 2026-06-14_11-48-48）",
  )
  parser.add_argument(
    "--wandb-run",
    dest="wandb_run_path",
    help="完整 W&B run 路径: entity/project/run_id",
  )
  parser.add_argument(
    "--entity",
    help="W&B entity（默认: WANDB_ENTITY / WANDB_USERNAME / wandb 默认账号）",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    help="自定义输出目录（默认 logs/rsl_rl/<experiment>/<run_name>/）",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)

  experiment = args.experiment
  run_name = args.run_name

  if args.wandb_run_path is None and run_name is None and experiment is not None:
    if RUN_ID_RE.match(experiment):
      run_name = experiment
      experiment = None

  try:
    download_run(
      experiment=experiment,
      run_name=run_name,
      wandb_run_path=args.wandb_run_path,
      entity=args.entity,
      output_dir=args.output_dir,
    )
  except ModuleNotFoundError as exc:
    print(f"错误: {exc}", file=sys.stderr)
    return 1
  except ValueError as exc:
    print(f"错误: {exc}", file=sys.stderr)
    return 1
  except Exception as exc:
    if exc.__class__.__name__ == "CommError" and exc.__class__.__module__.startswith("wandb"):
      print(f"错误: W&B 请求失败（请先 wandb login）: {exc}", file=sys.stderr)
      return 1
    raise
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
