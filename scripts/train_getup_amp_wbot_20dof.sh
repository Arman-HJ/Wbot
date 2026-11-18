#!/usr/bin/env bash
# WBOT 20DoF Getup + AMP：Recovery expert + WGAN-GP。
#
# 用法:
#   ./scripts/train_getup_amp_wbot_20dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_20dof_getup_amp}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/wbot_20dof/recovery}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

echo "[INFO] WBOT 20DoF Getup+AMP (Recovery expert + WGAN-GP)"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
echo "[INFO] data=$AMP_DATA"

TRAIN_ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --amp-data-path "$AMP_DATA"
  --video "$VIDEO"
  --video-interval "$VIDEO_INTERVAL"
  --video-length "$VIDEO_LENGTH"
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi

exec python scripts/train.py Wbot-20DoF-Getup-AMP-Flat \
  "${TRAIN_ARGS[@]}"
