#!/usr/bin/env bash
# X2 AMP v2：velocity v1 任务奖励 + WGAN-GP（task/disc 0.5/0.5）。
#
# 用法:
#   ./scripts/train_amp_x2_wgan.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-20000}"
MAX_ITERS="${MAX_ITERS:-100001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-x2_amp_vel_wgan_20k_0505_nopose}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/x2/engineai_locomotion}"
AMP_DATASETS="${AMP_DATASETS:-locomotion_agibot_x2:1.0,locomotion_agibot_x2_sym:1.0}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

echo "[INFO] X2 AMP RL (velocity v1 task rewards + WGAN-GP)"
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

exec python scripts/train.py AgiBot-X2-AMP-v2-Flat \
  "${TRAIN_ARGS[@]}"
