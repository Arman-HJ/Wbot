#!/usr/bin/env bash
# Leg-only WBOT 12DoF walk/run/turn/side/stand AMP locomotion (same task as 20DoF).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30000}"
RUN_NAME="${RUN_NAME:-wbot_12dof}"
EXTRA_ARGS=("$@")

if [[ ! -f src/assets/amp_motions/wbot_12dof/walk_run_amp/walk_moving.npz ]]; then
  "$PYTHON" -m scripts.make_wbot_12dof_amp_dataset
fi

exec "$PYTHON" -u scripts/train_amp.py AMP-Walk-Run-Wbot-12DOF \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger tensorboard \
  --gpu-ids '[0]' \
  "${EXTRA_ARGS[@]}"
