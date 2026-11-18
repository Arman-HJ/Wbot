#!/usr/bin/env bash
# WBOT 20DoF kick：G1 式简单 BeyondMimic 奖励 + 自适应 RSI 采样（kick4）。
# PD 与 AMP 对齐（armature）；actor 含 projected_gravity；
# 瞬时 push 间隔 (1–3)s、幅度固定 stock；RSI/term 固定 stock（无课程）；
# action_rate=-0.1，action_smoothness (acc L2)=-0.1（函数对齐 AMP）。
#
# 用法:
#   ./scripts/train_kick_wbot_20dof.sh



set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_20dof_kick}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
VIDEO="${VIDEO:-False}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-360}"

# Headless video: with both glvnd vendors, EGL device 0 is Mesa's broken DRI device on
# the NVIDIA card and device 1 is llvmpipe (CPU). Load only NVIDIA; Mesa as fallback.
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

echo "[INFO] WBOT 20DoF Kick (G1-style simple imitation + adaptive sampling, kick3)"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME video=$VIDEO"
if [[ -n "$CHECKPOINT_FILE" ]]; then
  echo "[INFO] resume checkpoint: $CHECKPOINT_FILE"
fi

TRAIN_ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --video "$VIDEO"
  --video-interval "$VIDEO_INTERVAL"
  --video-length "$VIDEO_LENGTH"
  --video-width "$VIDEO_WIDTH"
  --video-height "$VIDEO_HEIGHT"
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi

exec python scripts/train.py Wbot-20Dof-Kick \
  "${TRAIN_ARGS[@]}"
