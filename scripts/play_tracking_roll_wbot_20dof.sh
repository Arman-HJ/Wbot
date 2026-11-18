#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CKPT="${CHECKPOINT_FILE:-}"
if [[ -z "$CKPT" ]]; then
  # Prefer newest run dir under roll_to_prone_tracking (if any remain locally).
  RUN_ROOT="$ROOT/logs/rsl_rl/wbot_20dof_roll_to_prone_tracking"
  CKPT="$(find "$RUN_ROOT" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "No roll-to-prone checkpoint found under logs/rsl_rl/wbot_20dof_roll_to_prone_tracking/." >&2
  echo "Pass CHECKPOINT_FILE=/path/to/model_*.pt or retrain via scripts/train_tracking_roll_wbot_20dof.sh" >&2
  exit 1
fi
exec "${PYTHON:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}" scripts/play.py Wbot-20Dof-Tracking-RollToProne --run "$CKPT" --num-envs 1 "$@"
