#!/usr/bin/env bash
# Wait for the current DAVIS chasefirst train to exit, then start shoot-phase resume.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/HRJ/miniconda3/envs/env_mjlab/bin/python}"
CHASE_RUN="${CHASE_RUN:-$ROOT/logs/rsl_rl/davis_shoot_wbot_20dof/2026-11-13_17-08-05_full_amp4096_chasefirst_scratch}"
MONITOR="$ROOT/logs/rsl_rl/davis_shoot_wbot_20dof/monitor"
LOGDIR="$ROOT/logs/rsl_rl/davis_shoot_wbot_20dof"
TMUX_SESSION="${TMUX_SESSION:-davis_shoot_gpu1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$MONITOR"
echo "[handoff] watching chase run: $CHASE_RUN"
echo "[handoff] started=$(date -Iseconds)" | tee -a "$MONITOR/handoff_shootphase.log"

# Prefer waiting on the live chasefirst train PID if present.
wait_for_train_exit() {
  local pid
  while true; do
    pid="$(pgrep -f 'train.py Davis-Shoot-Wbot-20DOF' | head -1 || true)"
    if [[ -z "${pid}" ]]; then
      echo "[handoff] no Davis-Shoot train running — proceed" | tee -a "$MONITOR/handoff_shootphase.log"
      return 0
    fi
    # Still the chasefirst job?
    if ps -p "$pid" -o args= 2>/dev/null | grep -q 'chasefirst_scratch'; then
      echo "[handoff] $(date -Iseconds) chasefirst pid=$pid still running" | tee -a "$MONITOR/handoff_shootphase.log"
      # Wait until that PID exits (poll; avoid blocking forever on wrong pid reuse).
      while kill -0 "$pid" 2>/dev/null; do
        sleep 60
      done
      echo "[handoff] $(date -Iseconds) chasefirst pid=$pid exited" | tee -a "$MONITOR/handoff_shootphase.log"
      sleep 5
      return 0
    fi
    # Some other Davis train is running — wait for GPU1/train slot to free.
    echo "[handoff] $(date -Iseconds) other Davis train pid=$pid; waiting" | tee -a "$MONITOR/handoff_shootphase.log"
    sleep 120
  done
}

wait_for_train_exit

CKPT="$(ls -t "$CHASE_RUN"/model_*.pt 2>/dev/null | grep -v play_export | head -1 || true)"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[handoff] ERROR: no checkpoint in $CHASE_RUN" | tee -a "$MONITOR/handoff_shootphase.log"
  exit 1
fi
echo "[handoff] using ckpt=$CKPT" | tee -a "$MONITOR/handoff_shootphase.log"

# Avoid double-start.
if pgrep -f 'train.py Davis-Shoot-Wbot-20DOF' >/dev/null; then
  echo "[handoff] ERROR: a Davis train is already running; abort shoot start" | tee -a "$MONITOR/handoff_shootphase.log"
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="full_amp4096_shootphase_from_chasefirst"
TMUX_LOG="$LOGDIR/tmux_${RUN_NAME}_gpu1_${STAMP}.log"

CMD="cd $ROOT && CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl DAVIS_SHOOT_PHASE=1 \
$PYTHON -u scripts/train.py Davis-Shoot-Wbot-20DOF \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 40000 \
  --agent.run-name ${RUN_NAME} \
  --agent.logger tensorboard \
  --gpu-ids '[0]' \
  --video False \
  --checkpoint-file ${CKPT} \
  2>&1 | tee ${TMUX_LOG}; echo EXIT=\$?; exec bash"

if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$TMUX_SESSION"
fi
tmux send-keys -t "$TMUX_SESSION" C-c 2>/dev/null || true
sleep 1
tmux send-keys -t "$TMUX_SESSION" "$CMD" Enter

echo "[handoff] launched shootphase in tmux $TMUX_SESSION" | tee -a "$MONITOR/handoff_shootphase.log"
echo "[handoff] log=$TMUX_LOG" | tee -a "$MONITOR/handoff_shootphase.log"
echo "$CKPT" > "$MONITOR/handoff_shootphase_ckpt.txt"
echo "$TMUX_LOG" > "$MONITOR/handoff_shootphase_tmux_log.txt"

# Emit agent wake so the session can acknowledge.
echo "AGENT_LOOP_TICK_davis_walkmix {\"prompt\":\"【DAVIS 射门续训已自动启动】chasefirst 已结束，已从最新 ckpt 启动 full_amp4096_shootphase_from_chasefirst（DAVIS_SHOOT_PHASE=1, max_iter=40000）。请确认 GPU1/tmux 正常、已加载 ckpt，并更新 monitor/LAST_CHECK.md。\"}"
