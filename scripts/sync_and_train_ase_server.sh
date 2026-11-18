#!/usr/bin/env bash
# 同步 unitree_rl_mjlab + rsl_rl 到训练服务器并后台启动 ASE smoke 训练。
#
# 用法:
#   ./scripts/sync_and_train_ase_server.sh
#   SERVER=10.233.10.1 NUM_ENVS=4096 ./scripts/sync_and_train_ase_server.sh
#   SYNC_ONLY=1 ./scripts/sync_and_train_ase_server.sh   # 只同步不训练

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
RUN_NAME="${RUN_NAME:-smoke_engai1}"

echo "[3/3] launch ASE smoke on ${SERVER} (nohup)"
ssh "${USER}@${SERVER}" bash -s <<EOF
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_mjlab
cd ${REMOTE_RL}/${PROJECT}
chmod +x scripts/train_ase_smoke_engineai.sh
export MUJOCO_GL=egl
export NUM_ENVS=${NUM_ENVS}
export MAX_ITERS=${MAX_ITERS}
export RUN_NAME=${RUN_NAME}
nohup ./scripts/train_ase_smoke_engineai.sh > logs/rsl_rl/_console/ase_smoke_\${RUN_NAME}.log 2>&1 &
echo "PID=\$!"
sleep 2
tail -n 30 logs/rsl_rl/_console/ase_smoke_\${RUN_NAME}.log || true
EOF

echo "[DONE] 查看日志: ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/rsl_rl/_console/ase_smoke_${RUN_NAME}.log'"
