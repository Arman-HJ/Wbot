#!/usr/bin/env bash
# DAVIS depth-only goal-directed shooting on wbot 20-DoF (mjlab).
#
# Sanity (few envs, 200 iters):
#   bash scripts/train_davis_shoot_wbot_20dof.sh --sanity
# Full (4096 envs / AMP / 20k iters, football_jax-scale):
#   bash scripts/train_davis_shoot_wbot_20dof.sh
#   GPU_IDS='[1]' bash scripts/train_davis_shoot_wbot_20dof.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/HRJ/miniconda3/envs/env_mjlab/bin/python}"

SANITY=0
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == "--sanity" ]]; then
    SANITY=1
  else
    EXTRA+=("$arg")
  fi
done

if [[ "$SANITY" -eq 1 ]]; then
  TASK="Davis-Shoot-Wbot-20DOF-Sanity"
  NUM_ENVS="${NUM_ENVS:-64}"
  MAX_ITERS="${MAX_ITERS:-200}"
  RUN_NAME="${RUN_NAME:-sanity}"
else
  TASK="Davis-Shoot-Wbot-20DOF"
  NUM_ENVS="${NUM_ENVS:-4096}"
  MAX_ITERS="${MAX_ITERS:-20000}"
  RUN_NAME="${RUN_NAME:-full_amp4096}"
fi

GPU_IDS="${GPU_IDS:-[0]}"
# RGB train videos (1280x720) stall collect; default off. VIDEO=1 to enable.
VIDEO="${VIDEO:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

ARGS=(
  "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERS"
  --agent.run-name "$RUN_NAME"
  --agent.logger tensorboard
  --gpu-ids "$GPU_IDS"
)
if [[ "$VIDEO" == "1" || "$VIDEO" == "true" || "$VIDEO" == "True" ]]; then
  ARGS+=(--video True)
else
  ARGS+=(--video False)
fi

exec "$PYTHON" -u scripts/train.py "${ARGS[@]}" "${EXTRA[@]}"
