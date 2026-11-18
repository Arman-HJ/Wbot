#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30001}"
RUN_NAME="${RUN_NAME:-prone_grounded_start}"
CONSOLE_LOG="$PROJECT_ROOT/logs/rsl_rl/_console/grounded_tracking_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_ROOT/logs/rsl_rl/_console"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup setsid "$TRAIN_PYTHON" -u scripts/train.py Wbot-20DoF-Getup-Tracking-Grounded-Start \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger tensorboard \
  --video True --video-interval 2000 --video-length 200 \
  > "$CONSOLE_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf 'TRAIN_PID=%s\nCONSOLE_LOG=%s\n' "$TRAIN_PID" "$CONSOLE_LOG"
printf '%s\n' "$TRAIN_PID" > "$PROJECT_ROOT/logs/rsl_rl/_console/grounded_tracking.pid"
printf '%s\n' "$CONSOLE_LOG" > "$PROJECT_ROOT/logs/rsl_rl/_console/grounded_tracking.logpath"
