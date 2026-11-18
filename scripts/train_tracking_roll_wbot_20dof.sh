#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${PYTHON:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}" scripts/train.py Wbot-20Dof-Tracking-RollToProne \
  --env.scene.num-envs "${NUM_ENVS:-1024}" \
  --agent.max-iterations "${MAX_ITERS:-5001}" \
  --agent.save-interval "${SAVE_INTERVAL:-250}" \
  --agent.run-name "${RUN_NAME:-right_foot_fixed}" \
  --agent.logger tensorboard \
  --video "${VIDEO:-True}" \
  --video-interval "${VIDEO_INTERVAL:-2000}" \
  --video-length "${VIDEO_LENGTH:-150}" "$@"
