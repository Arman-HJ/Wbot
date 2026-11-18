#!/usr/bin/env bash
# SMP 阶段2（X2）：冻结先验 + PPO（velocity v1 任务奖励 + SDS 风格奖励）。
#
# 用法:
#   ./scripts/train_smp_policy_x2.sh
#   SMP_PRIOR_DIR=logs/smp_prior/x2_engineai_prior_env_100k \
#     SMP_PRIOR_MODEL=model_99999.pt \
#     ./scripts/train_smp_policy_x2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-100001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-x2_smp_rl_vel_4k_style003_nocurr}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/x2/engineai_locomotion}"
AMP_DATASETS="${AMP_DATASETS:-locomotion_agibot_x2:1.0,locomotion_agibot_x2_sym:1.0}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/x2_engineai_prior_env_100k}"
SMP_PRIOR_MODEL_NAME="${SMP_PRIOR_MODEL:-model_99999.pt}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-4000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.97}"
SMP_REWARD_WEIGHT="${SMP_REWARD_WEIGHT:-0.03}"
SDS_LOSS_SCALE="${SDS_LOSS_SCALE:-6.0}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
PRIOR_CFG="$SMP_PRIOR_DIR/diffusion_config.yaml"
PRIOR_MODEL="$SMP_PRIOR_DIR/$SMP_PRIOR_MODEL_NAME"

if [[ ! -f "$PRIOR_MODEL" ]]; then
  for cand in model_49999.pt model_50000.pt model_51999.pt model.pt; do
    if [[ -f "$SMP_PRIOR_DIR/$cand" ]]; then
      PRIOR_MODEL="$SMP_PRIOR_DIR/$cand"
      echo "[WARN] using fallback prior: $PRIOR_MODEL"
      break
    fi
  done
fi

if [[ "$(python - <<PY
w=float("${SMP_REWARD_WEIGHT}")
print(1 if w>0 else 0)
PY
)" == "1" ]]; then
  if [[ ! -f "$PRIOR_CFG" || ! -f "$PRIOR_MODEL" ]]; then
    echo "[ERROR] 缺少先验，请先跑先验训练" >&2
    echo "  cfg=$PRIOR_CFG model=$PRIOR_MODEL" >&2
    exit 1
  fi
fi

if [[ -z "${AMP_DATASETS:-}" ]]; then
  AMP_DATASETS="$(
    python - <<'PY' "$AMP_DATA"
from pathlib import Path
import sys
root = Path(sys.argv[1])
stems = sorted({p.stem for p in root.glob("*.npz")})
print(",".join(f"{s}:1.0" for s in stems))
PY
  )"
fi

echo "[INFO] X2 SMP policy RL (velocity v1 task rewards + SDS style)"
echo "[INFO] envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
echo "[INFO] reward mix: task=$TASK_REWARD_WEIGHT style=$SMP_REWARD_WEIGHT sds_loss_scale=$SDS_LOSS_SCALE"
if [[ -n "$CHECKPOINT_FILE" ]]; then
  echo "[INFO] resume checkpoint: $CHECKPOINT_FILE"
fi
if [[ "$(python - <<PY
w=float("${SMP_REWARD_WEIGHT}")
print(1 if w>0 else 0)
PY
)" == "1" ]]; then
  echo "[INFO] Prior (frozen): $PRIOR_MODEL"
fi
echo "[INFO] video=$VIDEO interval=$VIDEO_INTERVAL length=$VIDEO_LENGTH"

TRAIN_ARGS=(
  --env.scene.num-envs "$NUM_ENVS"
  --amp-data-path "$AMP_DATA"
  --amp-datasets "$AMP_DATASETS"
  --video "$VIDEO"
  --video-interval "$VIDEO_INTERVAL"
  --video-length "$VIDEO_LENGTH"
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
  --agent.algorithm.task-reward-weight "$TASK_REWARD_WEIGHT"
  --agent.algorithm.smp-reward-weight "$SMP_REWARD_WEIGHT"
  --agent.algorithm.sds-loss-scale "$SDS_LOSS_SCALE"
)
if [[ -n "$CHECKPOINT_FILE" ]]; then
  TRAIN_ARGS+=(--checkpoint_file "$CHECKPOINT_FILE")
fi
if [[ "$(python - <<PY
w=float("${SMP_REWARD_WEIGHT}")
print(1 if w>0 else 0)
PY
)" == "1" ]]; then
  TRAIN_ARGS+=(
    --smp-prior-cfg "$PRIOR_CFG"
    --smp-prior-model "$PRIOR_MODEL"
  )
fi

exec python scripts/train.py AgiBot-X2-SMP-Flat \
  "${TRAIN_ARGS[@]}"
