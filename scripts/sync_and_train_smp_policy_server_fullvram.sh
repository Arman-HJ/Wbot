#!/usr/bin/env bash
# 在指定服务器上跑 SMP RL，尽量拉满 GPU 显存（自动探测 NUM_ENVS）。
#
# 用法:
#   SERVER=172.16.0.13 ./scripts/sync_and_train_smp_policy_server_fullvram.sh
#   SERVER=10.233.10.1 NUM_ENVS=16384 ./scripts/sync_and_train_smp_policy_server_fullvram.sh
#
# 默认: 50k 先验 + amp_v2 任务奖励 (0.6) + SMP (0.4)

set -euo pipefail

SERVER="${SERVER:-172.16.0.13}"
USER="${USER:-songhat}"
REMOTE_RL="/home/songhat/code_workspace/rl"
LOCAL_RL="/home/songhat/code_workspace/rl"
PROJECT="unitree_rl_mjlab"
# 先验默认从本机或旧服务器已有目录同步；也可 PRIOR_SRC=host:path
PRIOR_SRC="${PRIOR_SRC:-songhat@10.233.10.1:${REMOTE_RL}/${PROJECT}/logs/smp_prior/engineai_prior_200k}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/engineai_prior_200k}"
SMP_PRIOR_MODEL="${SMP_PRIOR_MODEL:-model_49999.pt}"
MAX_ITERS="${MAX_ITERS:-5000}"
RUN_NAME="${RUN_NAME:-smp_rl_50kprior_fullvram}"
# 0 = 按显存自动估算
NUM_ENVS="${NUM_ENVS:-0}"
# 目标显存占用比例（留一点给驱动/碎片）
VRAM_FRAC="${VRAM_FRAC:-0.90}"
# 参考：RTX5090 上 4096 env ≈ 5.3 GiB → ~1.3 MiB/env（含 SMP SDS）
MIB_PER_ENV="${MIB_PER_ENV:-1.4}"

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

echo "[0] probe SSH ${USER}@${SERVER}"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${USER}@${SERVER}" 'echo ok' >/dev/null; then
  echo "[ERROR] 无法 SSH 到 ${SERVER}（握手失败或拒绝）。请检查：" >&2
  echo "  - 机器在线 / sshd 正常" >&2
  echo "  - 本机 IP 是否在 allowlist" >&2
  echo "  - 是否需要跳板: ssh -J jump ${USER}@${SERVER}" >&2
  exit 1
fi

echo "[1/4] rsync rsl_rl -> ${SERVER}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/rsl_rl/" "${USER}@${SERVER}:${REMOTE_RL}/rsl_rl/"

echo "[2/4] rsync ${PROJECT} -> ${SERVER}"
rsync -avz "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_RL/$PROJECT/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/"

echo "[3/4] sync 50k prior from ${PRIOR_SRC}"
ssh "${USER}@${SERVER}" "mkdir -p ${REMOTE_RL}/${PROJECT}/${SMP_PRIOR_DIR}"
# PRIOR_SRC 可以是本地目录或 user@host:path
if [[ "${PRIOR_SRC}" == *:* && "${PRIOR_SRC}" != /* ]]; then
  # remote → remote via this machine as relay, or pull then push
  TMP_PRIOR="$(mktemp -d)"
  rsync -avz "${PRIOR_SRC}/" "${TMP_PRIOR}/" \
    --include='diffusion_config.yaml' --include='env_config.yaml' \
    --include="${SMP_PRIOR_MODEL}" --include='model.pt' --exclude='*'
  rsync -avz "${TMP_PRIOR}/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${SMP_PRIOR_DIR}/"
  rm -rf "${TMP_PRIOR}"
else
  rsync -avz "${PRIOR_SRC}/" "${USER}@${SERVER}:${REMOTE_RL}/${PROJECT}/${SMP_PRIOR_DIR}/" \
    --include='diffusion_config.yaml' --include='env_config.yaml' \
    --include="${SMP_PRIOR_MODEL}" --include='model.pt' --exclude='*'
fi

echo "[4/4] launch full-VRAM SMP RL on ${SERVER}"
ssh "${USER}@${SERVER}" bash -s <<EOF
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_mjlab
cd ${REMOTE_RL}/${PROJECT}
chmod +x scripts/train_smp_policy_engineai.sh
mkdir -p logs/rsl_rl/_console

# 停掉本机旧的 SMP RL（如有）
pkill -f 'train.py AdamSP-SMP' || true
sleep 1

TOTAL_MIB=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
USED_MIB=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
FREE_MIB=\$(( TOTAL_MIB - USED_MIB))
echo "[GPU] total=\${TOTAL_MIB}MiB used=\${USED_MIB}MiB free=\${FREE_MIB}MiB"

NUM_ENVS=${NUM_ENVS}
if [[ "\$NUM_ENVS" -le 0 ]]; then
  TARGET_MIB=\$(python - <<PY
total=float(${TOTAL_MIB})
frac=float(${VRAM_FRAC})
mib_per=float(${MIB_PER_ENV})
budget=max(0.0, total*frac)
n=int(budget/mib_per)
# 对齐到 256，且至少 4096
n=max(4096, (n//256)*256)
print(n)
PY
)
  NUM_ENVS=\$TARGET_MIB
fi
echo "[INFO] NUM_ENVS=\$NUM_ENVS (VRAM target frac=${VRAM_FRAC}, ~${MIB_PER_ENV} MiB/env)"

export MUJOCO_GL=egl
export NUM_ENVS
export MAX_ITERS=${MAX_ITERS}
export RUN_NAME=${RUN_NAME}
export SMP_PRIOR_DIR=${SMP_PRIOR_DIR}
export SMP_PRIOR_MODEL=${SMP_PRIOR_MODEL}
nohup ./scripts/train_smp_policy_engineai.sh > logs/rsl_rl/_console/smp_policy_\${RUN_NAME}.log 2>&1 &
echo "PID=\$!"
sleep 8
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
tail -n 40 logs/rsl_rl/_console/smp_policy_\${RUN_NAME}.log || true
EOF

echo "[DONE] ssh ${USER}@${SERVER} 'tail -f ${REMOTE_RL}/${PROJECT}/logs/rsl_rl/_console/smp_policy_${RUN_NAME}.log'"
