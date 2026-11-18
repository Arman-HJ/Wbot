#!/usr/bin/env bash
# WBOT 20DoF tracking：e1_mimic victory2（配置对齐 Kick，无球）。
#
# 用法:
#   ./scripts/train_tracking_victory2_wbot_20dof.sh
#   NUM_ENVS=2048 RUN_NAME=victory2_v1 ./scripts/train_tracking_victory2_wbot_20dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-victory2}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
VIDEO="${VIDEO:-False}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-550}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-11.0}"
MOTION_FILE="${MOTION_FILE:-$ROOT/src/assets/motions/wbot_20dof/e1_mimic/dance/victory2_tracking.npz}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
  if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json && -r /dev/nvidiactl && -w /dev/nvidiactl ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  elif [[ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
    echo "[INFO] Using Mesa llvmpipe EGL (CPU) for headless video"
  fi
fi

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "[ERROR] motion file not found: $MOTION_FILE"
  exit 1
fi

echo "[INFO] WBOT 20DoF Tracking Victory2 (kick-aligned, no ball)"
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

exec python scripts/train.py Wbot-20Dof-Tracking-Victory2 \
  "${TRAIN_ARGS[@]}"
