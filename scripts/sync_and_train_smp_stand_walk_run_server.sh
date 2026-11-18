#!/usr/bin/env bash
# robot_stand_walk_run：同步数据 → 后台跑先验 100k → 自动接 SMP RL。
set -euo pipefail

SERVER="${SERVER:-10.233.10.1}"
USER="${USER:-songhat}"
REMOTE_RL="/home/songhat/code_workspace/rl"
LOCAL_RL="/home/songhat/code_workspace/rl"
PROJECT="unitree_rl_mjlab"

AMP_DATA="${AMP_DATA:-src/assets/amp_motions/pnd_adam_sp/robot_stand_walk_run}"
AMP_DATASETS="${AMP_DATASETS:-walk-run_pnd_adam_sp:1.0,run-walk_pnd_adam_sp:1.0,stand-walk_pnd_adam_sp:0.5,walk-stand_pnd_adam_sp:0.5,run_man_pnd_adam_sp:1.0}"
PRIOR_RUN_NAME="${PRIOR_RUN_NAME:-stand_walk_run_prior_100k}"
RUN_NAME="${RUN_NAME:-smp_rl_stand_walk_run_20k_t05s05}"
TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.5}"
SMP_REWARD_WEIGHT="${SMP_REWARD_WEIGHT:-0.5}"

RSYNC_EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='logs/'
  --exclude='.git/'
  --exclude='wandb/'
  --exclude='*.egg-info/'
  --exclude='*.log'
  --exclude='doc/'
  --exclude='patent/'
  --exclude='src/assets/robots/adam_sp/meshes/z1_assets'
  --exclude='src/assets/robots/agibot_x2/x2_assets/'
)

echo "[1/4] rsync rsl_rl"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/rsl_rl/" "${USER}@${SERVER}:${REMOTE_RL}/rsl_rl/"

echo "[2/4] rsync ${PROJECT}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/$PROJECT/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/"

echo "[2.5/4] rsync motion npz"
LOCAL_MOTION="${LOCAL_RL}/${PROJECT}/${AMP_DATA}"
ssh "${USER}@${SERVER}" "mkdir -p ${REMOTE_RL}/${PROJECT}/${AMP_DATA}"
rsync -avz \
  --include='*.npz' --include='*.yaml' --include='*.json' --exclude='*' \
  "$LOCAL_MOTION/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${AMP_DATA}/"

if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
  echo "[DONE] sync only"
  exit 0
fi

echo "[3/4] stop old SMP jobs"
ssh "${USER}@${SERVER}" bash <<'EOS'
pgrep -f 'AdamSP-SMP-Flat' | xargs -r kill 2>/dev/null || true
pgrep -f 'train_smp_prior.py' | xargs -r kill 2>/dev/null || true
sleep 2
EOS

PIPE_LOG="logs/smp_prior/_console/${PRIOR_RUN_NAME}_pipeline.log"
echo "[4/4] launch pipeline (nohup) log=${PIPE_LOG}"
ssh "${USER}@${SERVER}" \
  "cd ${REMOTE_RL}/${PROJECT} && chmod +x scripts/run_smp_stand_walk_run_pipeline.sh && \
   nohup env AMP_DATA=${AMP_DATA} AMP_DATASETS='${AMP_DATASETS}' \
   PRIOR_RUN_NAME=${PRIOR_RUN_NAME} RUN_NAME=${RUN_NAME} \
   TASK_REWARD_WEIGHT=${TASK_REWARD_WEIGHT} SMP_REWARD_WEIGHT=${SMP_REWARD_WEIGHT} \
   RL_ONLY=${RL_ONLY:-0} PRIOR_ONLY=${PRIOR_ONLY:-0} \
   ./scripts/run_smp_stand_walk_run_pipeline.sh > ${PIPE_LOG} 2>&1 & echo PIPELINE_PID=\$!; sleep 8; tail -n 35 ${PIPE_LOG} || true"

echo "[DONE] tail -f ${REMOTE_RL}/${PROJECT}/${PIPE_LOG}"
