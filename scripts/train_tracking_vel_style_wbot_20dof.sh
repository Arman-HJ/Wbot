#!/usr/bin/env bash
# WBOT 20DoF velocity-conditioned style tracking (Driving-the-Style scaffold).
# Dense BeyondMimic rewards; actor command = (vx, vy, wz); no AMP disc.
#
# Prep paper-protocol motions (once):
#   python scripts/prepare_vel_style_paper_motions.py --also-mirror --also-smoothed
#
# Usage (generalization = train on random_walk only):
#   ./scripts/train_tracking_vel_style_wbot_20dof.sh
#   MOTION_FILE=.../side_tracking.npz NUM_ENVS=2048 ./scripts/train_tracking_vel_style_wbot_20dof.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-wbot_20dof_vel_style}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
MOTION_FILE="${MOTION_FILE:-}"
VIDEO="${VIDEO:-False}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-360}"

DEFAULT_MOTION="src/assets/motions/wbot_20dof/vel_style/random_walk_tracking.npz"
if [[ -z "$MOTION_FILE" ]]; then
  MOTION_FILE="$DEFAULT_MOTION"
fi
if [[ ! -f "$MOTION_FILE" ]]; then
  echo "[ERROR] Motion file missing: $MOTION_FILE"
  echo "[HINT] Build paper-protocol set first:"
  echo "  python scripts/prepare_vel_style_paper_motions.py --also-mirror --also-smoothed"
  exit 1
fi

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

echo "[INFO] WBOT 20DoF Tracking-VelStyle (velocity-conditioned BeyondMimic)"
echo "[INFO] motion=$MOTION_FILE envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"

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
  --motion-file "$MOTION_FILE"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi

exec python scripts/train.py Wbot-20Dof-Tracking-VelStyle \
  "${TRAIN_ARGS[@]}"
