#!/usr/bin/env bash
exec /home/songhat/miniconda3/envs/env_mjlab/bin/python -m tensorboard.main \
  --logdir /home/songhat/code_workspace/unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_kick/2026-10-27_18-15-30_kick4_ampstand_ankle12mm_20261027_181526 \
  --host 0.0.0.0 --port 6017
