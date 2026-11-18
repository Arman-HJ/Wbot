#!/usr/bin/env bash
set -euo pipefail
MOTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$MOTION_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}"
export MUJOCO_GL=glfw
exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/play_wbot_dynamic_motion.py" --motion "$MOTION_DIR/rollout_motion.npz" --physics "$MOTION_DIR/physics_rollout.npz" --model "$MOTION_DIR/simulation_model.mjb" --hold-final "$@"
