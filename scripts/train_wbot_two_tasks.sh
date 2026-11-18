#!/usr/bin/env bash
set -euo pipefail
cd /home/songhat/code_workspace/unitree_rl_mjlab
PYTHON=/home/songhat/miniconda3/envs/env_mjlab/bin/python
RUN_DIR=logs/launch_20261002
mkdir -p "$RUN_DIR"
for task in Wbot-12Dof-Flat Wbot-12Dof-Suspended; do
  if [[ -f "$RUN_DIR/$task.pid" ]] && kill -0 "$(cat "$RUN_DIR/$task.pid")" 2>/dev/null; then
    continue
  fi
  logfile="$RUN_DIR/$task.log"
  case "$task" in
    Wbot-12Dof-Flat) run_name=walk_reference_xml_standvel005_v5 ;;
    Wbot-12Dof-Suspended) run_name=suspended_reference_xml_delay20ms_v5 ;;
  esac
  setsid nohup "$PYTHON" -u scripts/train.py "$task" --env.scene.num-envs 2048 --agent.max-iterations 10001 --agent.logger tensorboard --agent.run-name "$run_name" --video True --video-interval 2000 --video-length 200 --video-width 640 --video-height 360 > "$logfile" 2>&1 < /dev/null &
  echo "$!" > "$RUN_DIR/$task.pid"
done
for pair in 'wbot_12dof_velocity 6011' 'wbot_12dof_suspended 6012'; do
  read -r experiment port <<< "$pair"
  if [[ -f "$RUN_DIR/tensorboard_$port.pid" ]] && kill -0 "$(cat "$RUN_DIR/tensorboard_$port.pid")" 2>/dev/null; then
    continue
  fi
  setsid nohup "$PYTHON" -m tensorboard.main --logdir "logs/rsl_rl/$experiment" --host 127.0.0.1 --port "$port" > "$RUN_DIR/tensorboard_$port.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN_DIR/tensorboard_$port.pid"
done
