#!/usr/bin/env bash
# 在训练服务器本地执行：robot_stand_walk_run 先验 → SMP RL。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
  source ~/miniforge3/etc/profile.d/conda.sh
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
else
  echo "[ERROR] conda not found" >&2
  exit 1
fi
conda activate env_mjlab
export MUJOCO_GL=egl

AMP_DATA="${AMP_DATA:-src/assets/amp_motions/pnd_adam_sp/robot_stand_walk_run}"
AMP_DATASETS="${AMP_DATASETS:-walk-run_pnd_adam_sp:1.0,run-walk_pnd_adam_sp:1.0,stand-walk_pnd_adam_sp:0.5,walk-stand_pnd_adam_sp:0.5,run_man_pnd_adam_sp:1.0}"

PRIOR_ITERS="${PRIOR_ITERS:-100000}"
PRIOR_RUN_NAME="${PRIOR_RUN_NAME:-stand_walk_run_prior_100k}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/${PRIOR_RUN_NAME}}"

NUM_ENVS="${NUM_ENVS:-20000}"
MAX_ITERS="${MAX_ITERS:-100001}"
RL_RUN_NAME="${RL_RUN_NAME:-${RUN_NAME:-smp_rl_stand_walk_run_20k_t05s05}}"
TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.5}"
SMP_REWARD_WEIGHT="${SMP_REWARD_WEIGHT:-0.5}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

RL_ONLY="${RL_ONLY:-0}"
PRIOR_ONLY="${PRIOR_ONLY:-0}"

chmod +x scripts/train_smp_prior_engineai.sh scripts/train_smp_policy_engineai.sh
mkdir -p logs/smp_prior/_console logs/rsl_rl/_console

if [[ "${RL_ONLY}" != "1" ]]; then
  echo "========== [PRIOR] ${PRIOR_RUN_NAME} =========="
  export AMP_DATA AMP_DATASETS PRIOR_ITERS SMP_PRIOR_DIR
  export RUN_NAME="${PRIOR_RUN_NAME}"
  export DEVICE="${DEVICE:-cuda:0}"
  export LOG_ITER="${LOG_ITER:-200}"
  ./scripts/train_smp_prior_engineai.sh
  echo "[PRIOR DONE] dir=${SMP_PRIOR_DIR}"
fi

if [[ "${PRIOR_ONLY}" == "1" ]]; then
  echo "[DONE] prior only"
  exit 0
fi

PRIOR_MODEL_PATH="${SMP_PRIOR_DIR}/model_99999.pt"
[[ -f "$PRIOR_MODEL_PATH" ]] || PRIOR_MODEL_PATH="${SMP_PRIOR_DIR}/model.pt"
if [[ ! -f "$PRIOR_MODEL_PATH" ]]; then
  echo "[ERROR] prior model missing under ${SMP_PRIOR_DIR}" >&2
  ls -la "${SMP_PRIOR_DIR}"/model_*.pt 2>/dev/null | tail -10 >&2 || true
  exit 1
fi

echo "========== [RL] ${RL_RUN_NAME} =========="
export AMP_DATA AMP_DATASETS NUM_ENVS MAX_ITERS RUN_NAME="${RL_RUN_NAME}" SMP_PRIOR_DIR
export SMP_PRIOR_MODEL="$(basename "$PRIOR_MODEL_PATH")"
export TASK_REWARD_WEIGHT SMP_REWARD_WEIGHT VIDEO VIDEO_INTERVAL VIDEO_LENGTH
nohup ./scripts/train_smp_policy_engineai.sh > "logs/rsl_rl/_console/smp_policy_${RL_RUN_NAME}.log" 2>&1 &
echo "RL PID=$!"
