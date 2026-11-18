#!/usr/bin/env bash
# 同步代码并在服务器上只跑 SMP 先验预训练（自监督，无 RL）。
#
# 用法:
#   ./scripts/sync_and_train_smp_prior_server.sh
#   PRIOR_ITERS=100000 RUN_NAME=engineai_prior1 ./scripts/sync_and_train_smp_prior_server.sh
#   SYNC_ONLY=1 ./scripts/sync_and_train_smp_prior_server.sh

set -euo pipefail

SERVER="${SERVER:-10.233.10.1}"
USER="${USER:-songhat}"
REMOTE_RL="/home/songhat/code_workspace/rl"
LOCAL_RL="/home/songhat/code_workspace/rl"
PROJECT="unitree_rl_mjlab"

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

echo "[1/3] rsync rsl_rl -> ${USER}@${SERVER}:${REMOTE_RL}/rsl_rl"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/rsl_rl/" "${USER}@${SERVER}:${REMOTE_RL}/rsl_rl/"

echo "[2/3] rsync ${PROJECT} -> ${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/$PROJECT/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/"

echo "[2.5/3] rsync motion npz -> ${AMP_DATA}"
LOCAL_MOTION="${LOCAL_RL}/${PROJECT}/${AMP_DATA}"
ssh "${USER}@${SERVER}" "mkdir -p ${REMOTE_RL}/${PROJECT}/${AMP_DATA}"
rsync -avz \
  --include='*.npz' --include='*.yaml' --include='*.json' --exclude='*' \
  "$LOCAL_MOTION/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${AMP_DATA}/"

if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
  echo "[DONE] sync only"
  exit 0
fi

PRIOR_ITERS="${PRIOR_ITERS:-100000}"
RUN_NAME="${RUN_NAME:-engineai_prior_env_100k_$(date +%Y%m%d_%H%M%S)}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/pnd_adam_sp/engineai_locomotion}"
AMP_DATASETS="${AMP_DATASETS:-}"
ENV_TASK="${ENV_TASK:-AdamSP-SMP-Flat}"
PRIOR_CFG="${PRIOR_CFG:-src/tasks/smp/configs/tinymdm_adam_sp.yaml}"
DEVICE="${DEVICE:-cuda:0}"
LOG_ITER="${LOG_ITER:-200}"

echo "[3/3] launch SMP prior-only on ${SERVER} (nohup)"
ssh "${USER}@${SERVER}" bash -s <<EOF
set -euo pipefail
if [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
  source ~/miniforge3/etc/profile.d/conda.sh
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
else
  echo "[ERROR] conda not found" >&2
  exit 1
fi
conda activate env_mjlab
cd ${REMOTE_RL}/${PROJECT}
chmod +x scripts/train_smp_prior_engineai.sh scripts/train_smp_prior.py
mkdir -p logs/smp_prior/_console
export MUJOCO_GL=egl
export PRIOR_ITERS=${PRIOR_ITERS}
export RUN_NAME=${RUN_NAME}
export DEVICE=${DEVICE}
export LOG_ITER=${LOG_ITER}
export AMP_DATA=${AMP_DATA}
export AMP_DATASETS='${AMP_DATASETS}'
export ENV_TASK=${ENV_TASK}
export PRIOR_CFG=${PRIOR_CFG}
# 固定 out dir 与 run name 一致，便于查看
export SMP_PRIOR_DIR=logs/smp_prior/\${RUN_NAME}
nohup ./scripts/train_smp_prior_engineai.sh > logs/smp_prior/_console/\${RUN_NAME}_launch.log 2>&1 &
echo "PID=\$!"
echo "RUN_NAME=\${RUN_NAME}"
echo "PRIOR_DIR=logs/smp_prior/\${RUN_NAME}"
sleep 4
tail -n 50 logs/smp_prior/_console/\${RUN_NAME}.log 2>/dev/null \
  || tail -n 50 logs/smp_prior/_console/\${RUN_NAME}_launch.log || true
EOF

echo "[DONE] 查看日志:"
echo "  ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/smp_prior/_console/${RUN_NAME}.log'"
echo "  tensorboard: logs/smp_prior/${RUN_NAME}"
