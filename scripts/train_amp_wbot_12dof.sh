#!/usr/bin/env bash
# WBOT 12DoF AMP v2：velocity 任务奖励 + WGAN-GP（task/disc 0.5/0.5）。
#
# 用法:
#   ./scripts/train_amp_wbot_12dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-100001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_12dof_amp_v2}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/wbot_12dof/engineai_locomotion}"
AMP_DATASETS="${AMP_DATASETS:-locomotion_wbot_12dof:1.0,locomotion_wbot_12dof_sym:1.0}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

echo "[INFO] WBOT 12DoF AMP v2 (velocity task rewards + WGAN-GP)"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
echo "[INFO] reward mix: task=0.5 disc=0.5"
echo "[INFO] data=$AMP_DATA datasets=$AMP_DATASETS"
if [[ -n "$CHECKPOINT_FILE" ]]; then
  echo "[INFO] resume checkpoint: $CHECKPOINT_FILE"
fi

TRAIN_ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --amp-data-path "$AMP_DATA"
  --amp-datasets "$AMP_DATASETS"
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

exec python scripts/train.py Wbot-12Dof-AMP-v2-Flat \
  "${TRAIN_ARGS[@]}"
