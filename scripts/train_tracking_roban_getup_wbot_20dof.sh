#!/usr/bin/env bash
# WBOT 20DoF 纯 tracking：Roban 右脚前踏起身（step_getup，50 FPS）。
#
# 用法:
#   ./scripts/train_tracking_roban_getup_wbot_20dof.sh
#   NUM_ENVS=2048 ./scripts/train_tracking_roban_getup_wbot_20dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_20dof_getup_priv}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
# 17s clip at 50 FPS; 950 steps covers the clip plus slack.
VIDEO_LENGTH="${VIDEO_LENGTH:-950}"
# Clip = 17s at 50Hz; leave slack before timeout.
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-19.0}"
MOTION_FILE="${MOTION_FILE:-$ROOT/src/assets/motions/wbot_20dof/getup/prone_1_robans221_1011_wbot20.npz}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "[ERROR] motion file not found: $MOTION_FILE"
  exit 1
fi

echo "[INFO] WBOT 20DoF Roban Step-Getup Tracking (adaptive + video + tensorboard + EE)"
echo "[INFO] motion=$MOTION_FILE"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS episode=${EPISODE_LENGTH_S}s run=$RUN_NAME video=$VIDEO"
if [[ -n "$CHECKPOINT_FILE" ]]; then
  echo "[INFO] resume checkpoint: $CHECKPOINT_FILE"
fi

TRAIN_ARGS=(
  --motion-file "$MOTION_FILE"
  --env.commands.motion.sampling-mode adaptive
  --env.episode-length-s "$EPISODE_LENGTH_S"
  --env.scene.num-envs "$NUM_ENVS"
  --video "$VIDEO"
  --video-interval "$VIDEO_INTERVAL"
  --video-length "$VIDEO_LENGTH"
  --agent.logger tensorboard
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi

exec python scripts/train.py Wbot-20Dof-Tracking-Getup \
  "${TRAIN_ARGS[@]}"
