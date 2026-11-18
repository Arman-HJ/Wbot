#!/usr/bin/env bash
# SMP 阶段1：自监督预训练 TinyMDM motion prior（无仿真、无 PPO）。
#
# 日志目录（带时间戳）:
#   logs/smp_prior/<run_name>/
#     log.txt  metrics.csv  events.out.tfevents.*  model.pt  diffusion_config.yaml
#
# 用法:
#   ./scripts/train_smp_prior_engineai.sh
#   PRIOR_ITERS=100000 ./scripts/train_smp_prior_engineai.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AMP_DATA="${AMP_DATA:-src/assets/amp_motions/pnd_adam_sp/engineai_locomotion}"
ENV_TASK="${ENV_TASK:-AdamSP-SMP-Flat}"
PRIOR_CFG="${PRIOR_CFG:-src/tasks/smp/configs/tinymdm_adam_sp.yaml}"
RUN_NAME="${RUN_NAME:-engineai_$(date +%Y%m%d_%H%M%S)}"
SMP_PRIOR_DIR="${SMP_PRIOR_DIR:-logs/smp_prior/${RUN_NAME}}"
PRIOR_ITERS="${PRIOR_ITERS:-100000}"
DEVICE="${DEVICE:-cuda:0}"
LOG_ITER="${LOG_ITER:-200}"

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

mkdir -p "$SMP_PRIOR_DIR" logs/smp_prior/_console
CONSOLE_LOG="logs/smp_prior/_console/${RUN_NAME}.log"

echo "[INFO] SMP prior (self-supervised) only — no robot RL"
echo "[INFO] data=$AMP_DATA datasets=$AMP_DATASETS env_task=$ENV_TASK"
echo "[INFO] cfg=$PRIOR_CFG out=$SMP_PRIOR_DIR iters=$PRIOR_ITERS device=$DEVICE"
echo "[INFO] console log: $CONSOLE_LOG"

python scripts/train_smp_prior.py \
  --cfg "$PRIOR_CFG" \
  --out-dir "$SMP_PRIOR_DIR" \
  --amp-data-path "$AMP_DATA" \
  --amp-datasets "$AMP_DATASETS" \
  --env-task "$ENV_TASK" \
  --device "$DEVICE" \
  --num-iterations "$PRIOR_ITERS" \
  --log-iter "$LOG_ITER" \
  2>&1 | tee "$CONSOLE_LOG"

echo "[DONE] prior dir: $SMP_PRIOR_DIR"
echo "[DONE] tensorboard --logdir $SMP_PRIOR_DIR"
