#!/usr/bin/env bash
# Play / record DAVIS shooting policy on wbot.
#
# Interactive:
#   bash scripts/play_davis_shoot_wbot_20dof.sh --run <run_dir_or_name>
# Headless video:
#   bash scripts/play_davis_shoot_wbot_20dof.sh --run <run> --video --video-length 300
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/HRJ/miniconda3/envs/env_mjlab/bin/python}"
TASK="${TASK:-Davis-Shoot-Wbot-20DOF}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$PYTHON" -u scripts/play.py "$TASK" "$@"
