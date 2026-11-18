#!/usr/bin/env bash
# WBOT 12DoF kick：动作模仿（无 AMP）+ 前方放球，球前进 3m 得分。
#
# 用法:
#   ./scripts/train_kick_wbot_12dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_12dof_kick}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

echo "[INFO] WBOT 12DoF Kick (tracking + ball, no AMP, GVHMR kick2)"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
if [[ -n "$CHECKPOINT_FILE" ]]; then
  echo "[INFO] resume checkpoint: $CHECKPOINT_FILE"
fi

TRAIN_ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi

exec python scripts/train.py Wbot-12Dof-Kick "${TRAIN_ARGS[@]}"
