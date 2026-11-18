#!/usr/bin/env bash
# 同步并启动 X2 SMP 策略 RL（velocity v1 任务奖励 + x2 先验）。
#
# 用法:
#   ./scripts/sync_and_train_smp_policy_x2_server.sh
#   RUN_NAME=x2_smp_rl_vel_16k ./scripts/sync_and_train_smp_policy_x2_server.sh

set -euo pipefail

SERVER="${SERVER:-10.233.10.1}"
USER="${USER:-songhat}"
REMOTE_RL="/home/songhat/code_workspace/rl"
LOCAL_RL="/home/songhat/code_workspace/rl"
PROJECT="unitree_rl_mjlab"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-100001}"
RUN_NAME="${RUN_NAME:-x2_smp_rl_vel_4k_style003_nocurr}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/x2_engineai_prior_env_100k}"
SMP_PRIOR_MODEL="${SMP_PRIOR_MODEL:-model_99999.pt}"
TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.97}"
SMP_REWARD_WEIGHT="${SMP_REWARD_WEIGHT:-0.03}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
SDS_LOSS_SCALE="${SDS_LOSS_SCALE:-6.0}"
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

echo "[2.5/3] rsync SMP prior -> ${SERVER}:${SMP_PRIOR_DIR}"
if [[ "$(python - <<PY
w=float("${SMP_REWARD_WEIGHT:-0.2}")
print(1 if w>0 else 0)
PY
)" == "1" ]]; then
LOCAL_PRIOR="${LOCAL_PRIOR:-$LOCAL_RL/$PROJECT/$SMP_PRIOR_DIR}"
REMOTE_PRIOR="$SMP_PRIOR_DIR"
if [[ ! -f "$LOCAL_PRIOR/$SMP_PRIOR_MODEL" ]]; then
  echo "[WARN] local prior missing: $LOCAL_PRIOR/$SMP_PRIOR_MODEL — will use server copy if present"
else
  ssh "${USER}@${SERVER}" "mkdir -p ${REMOTE_RL}/${PROJECT}/${REMOTE_PRIOR}"
  rsync -avz \
    "$LOCAL_PRIOR/diffusion_config.yaml" \
    "$LOCAL_PRIOR/env_config.yaml" \
    "$LOCAL_PRIOR/$SMP_PRIOR_MODEL" \
    "$LOCAL_PRIOR/model.pt" \
    "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${REMOTE_PRIOR}/" 2>/dev/null || \
  rsync -avz \
    "$LOCAL_PRIOR/diffusion_config.yaml" \
    "$LOCAL_PRIOR/env_config.yaml" \
    "$LOCAL_PRIOR/$SMP_PRIOR_MODEL" \
    "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${REMOTE_PRIOR}/"
fi
else
  echo "[INFO] skip prior sync (SMP_REWARD_WEIGHT=${SMP_REWARD_WEIGHT:-0.2})"
fi

if [[ "${SYNC_ONLY:-0}" == "1" ]]; then
  echo "[DONE] sync only"
  exit 0
fi

echo "[3/3] stop old training + launch X2 SMP RL on ${SERVER}"
ssh "${USER}@${SERVER}" bash <<'EOS'
pgrep -f 'scripts/train.py' | xargs -r kill 2>/dev/null || true
pgrep -f 'AgiBot-X2' | xargs -r kill 2>/dev/null || true
pgrep -f 'AdamSP-SMP-Flat' | xargs -r kill 2>/dev/null || true
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
chmod +x scripts/train_smp_policy_x2.sh

PRIOR_DIR=${SMP_PRIOR_DIR}
PRIOR_MODEL=\$PRIOR_DIR/${SMP_PRIOR_MODEL}
if [[ "${SMP_REWARD_WEIGHT}" != "0" && "${SMP_REWARD_WEIGHT}" != "0.0" ]]; then
  if [[ ! -f "\$PRIOR_MODEL" ]]; then
    echo "[ERROR] missing \$PRIOR_MODEL" >&2
    ls -la "\$PRIOR_DIR"/model_*.pt 2>/dev/null | tail -20 >&2 || true
    exit 1
  fi
fi

mkdir -p logs/rsl_rl/_console
export MUJOCO_GL=egl
export NUM_ENVS=${NUM_ENVS}
export MAX_ITERS=${MAX_ITERS}
export RUN_NAME=${RUN_NAME}
export SMP_PRIOR_DIR=${SMP_PRIOR_DIR}
export SMP_PRIOR_MODEL=${SMP_PRIOR_MODEL}
export TASK_REWARD_WEIGHT=${TASK_REWARD_WEIGHT}
export SMP_REWARD_WEIGHT=${SMP_REWARD_WEIGHT}
export VIDEO=${VIDEO}
export VIDEO_INTERVAL=${VIDEO_INTERVAL}
export VIDEO_LENGTH=${VIDEO_LENGTH}
export SDS_LOSS_SCALE=${SDS_LOSS_SCALE}
export CHECKPOINT_FILE='${CHECKPOINT_FILE}'
nohup ./scripts/train_smp_policy_x2.sh > logs/rsl_rl/_console/smp_policy_\${RUN_NAME}.log 2>&1 &
echo "PID=\$!"
sleep 5
tail -n 40 logs/rsl_rl/_console/smp_policy_\${RUN_NAME}.log || true
EOF

echo "[DONE] ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/rsl_rl/_console/smp_policy_${RUN_NAME}.log'"
