#!/usr/bin/env bash
B=/home/songhat/code_workspace/unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_walk_run_amp
exec /home/songhat/miniconda3/envs/env_mjlab/bin/python -m tensorboard.main \
  --logdir_spec "r1800:$B/2026-10-27_14-28-06_ankle12mm_geom_only_resume1800_20261027_142802,r5000:$B/2026-10-27_15-45-04_ankle12mm_geom_only_resume5000_20261027_154459,r5900:$B/2026-10-27_17-08-26_ankle12mm_geom_only_resume5900_20261027_170822,r6400:$B/2026-10-27_18-14-09_ankle12mm_geom_only_resume6400_20261027_181405" \
  --host 0.0.0.0 --port 6016
