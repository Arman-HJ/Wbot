#!/usr/bin/env bash
# 同步 unitree_rl_mjlab + rsl_rl 到训练服务器并后台启动 SMP smoke（先验+策略）。
#
# 用法:
#   ./scripts/sync_and_train_smp_server.sh
#   SERVER=10.233.10.1 PRIOR_ITERS=20000 MAX_ITERS=5000 ./scripts/sync_and_train_smp_server.sh
#   SYNC_ONLY=1 ./scripts/sync_and_train_smp_server.sh
#   SKIP_PRIOR=1 ./scripts/sync_and_train_smp_server.sh

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
  --exclude='src/assets/robots/adam_sp/meshes/z1_assets'
  --exclude='src/assets/robots/agibot_x2/x2_assets/'
)

echo "[1/3] rsync rsl_rl -> ${USER}@${SERVER}:${REMOTE_RL}/rsl_rl"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/rsl_rl/" "${USER}@${SERVER}:${REMOTE_RL}/rsl_rl/"

echo "[2/3] rsync ${PROJECT} -> ${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/$PROJECT/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/"

if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
  echo "[DONE] sync only"
  exit 0
fi

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-5000}"
PRIOR_ITERS="${PRIOR_ITERS:-20000}"
SKIP_PRIOR="${SKIP_PRIOR:-0}"
RUN_POLICY="${RUN_POLICY:-0}"
RUN_NAME="${RUN_NAME:-smoke_smp1}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/engineai_smoke}"

echo "[3/3] launch SMP on ${SERVER} (default: prior-only; set RUN_POLICY=1 for RL)"
ssh "${USER}@${SERVER}" bash -s <<EOF
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_mjlab
cd ${REMOTE_RL}/${PROJECT}
chmod +x scripts/train_smp_smoke_engineai.sh scripts/train_smp_prior_engineai.sh scripts/train_smp_policy_engineai.sh
mkdir -p logs/rsl_rl/_console logs/smp_prior
export MUJOCO_GL=egl
export NUM_ENVS=${NUM_ENVS}
export MAX_ITERS=${MAX_ITERS}
export PRIOR_ITERS=${PRIOR_ITERS}
export SKIP_PRIOR=${SKIP_PRIOR}
export RUN_POLICY=${RUN_POLICY}
export RUN_NAME=${RUN_NAME}
export SMP_PRIOR_DIR=${SMP_PRIOR_DIR}
nohup ./scripts/train_smp_smoke_engineai.sh > logs/rsl_rl/_console/smp_smoke_\${RUN_NAME}.log 2>&1 &
echo "PID=\$!"
sleep 3
tail -n 40 logs/rsl_rl/_console/smp_smoke_\${RUN_NAME}.log || true
EOF

echo "[DONE] 查看日志: ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/rsl_rl/_console/smp_smoke_${RUN_NAME}.log'"
