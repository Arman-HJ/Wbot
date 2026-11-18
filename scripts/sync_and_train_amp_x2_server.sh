#!/usr/bin/env bash
# 同步并启动 X2 AMP（velocity v1 任务奖励 + WGAN 0.5/0.5），并停止 SMP。
#
# 用法:
#   ./scripts/sync_and_train_amp_x2_server.sh

set -euo pipefail

SERVER="${SERVER:-10.233.10.1}"
USER="${USER:-songhat}"
REMOTE_RL="/home/songhat/code_workspace/rl"
LOCAL_RL="/home/songhat/code_workspace/rl"
PROJECT="unitree_rl_mjlab"

NUM_ENVS="${NUM_ENVS:-20000}"
MAX_ITERS="${MAX_ITERS:-100001}"
RUN_NAME="${RUN_NAME:-x2_amp_vel_wgan_20k_0505_nopose}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

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

echo "[1/3] rsync rsl_rl"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/rsl_rl/" "${USER}@${SERVER}:${REMOTE_RL}/rsl_rl/"

echo "[2/3] rsync ${PROJECT}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/$PROJECT/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/"

if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
  echo "[DONE] sync only"
  exit 0
fi

echo "[3/3] stop SMP + launch X2 AMP on ${SERVER}"
ssh "${USER}@${SERVER}" bash <<'EOS'
pgrep -f 'AgiBot-X2-SMP-Flat' | xargs -r kill 2>/dev/null || true
pgrep -f 'train_smp_policy_x2' | xargs -r kill 2>/dev/null || true
pgrep -f 'scripts/train.py' | xargs -r kill 2>/dev/null || true
sleep 2
EOS
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
chmod +x scripts/train_amp_x2_wgan.sh

mkdir -p logs/rsl_rl/_console
export MUJOCO_GL=egl
export NUM_ENVS=${NUM_ENVS}
export MAX_ITERS=${MAX_ITERS}
export RUN_NAME=${RUN_NAME}
export VIDEO=${VIDEO}
export VIDEO_INTERVAL=${VIDEO_INTERVAL}
export VIDEO_LENGTH=${VIDEO_LENGTH}
export CHECKPOINT_FILE='${CHECKPOINT_FILE}'
nohup ./scripts/train_amp_x2_wgan.sh > logs/rsl_rl/_console/amp_x2_\${RUN_NAME}.log 2>&1 &
echo "PID=\$!"
sleep 5
tail -n 40 logs/rsl_rl/_console/amp_x2_\${RUN_NAME}.log || true
EOF

echo "[DONE] ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/rsl_rl/_console/amp_x2_${RUN_NAME}.log'"
