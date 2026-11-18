#!/usr/bin/env bash
# DroidUpE1-aligned WBOT 20DoF walk/run/turn/side/stand AMP locomotion.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/HRJ/miniconda3/envs/env_mjlab/bin/python}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30000}"
RUN_NAME="${RUN_NAME:-action_scale_025}"
GPU_IDS="${GPU_IDS:-[0]}"
EXTRA_ARGS=("$@")

exec "$PYTHON" -u scripts/train_amp.py AMP-Walk-Run-Wbot-20DOF \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger tensorboard \
  --gpu-ids "$GPU_IDS" \
  "${EXTRA_ARGS[@]}"
