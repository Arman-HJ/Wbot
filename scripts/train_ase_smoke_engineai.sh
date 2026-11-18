#!/usr/bin/env bash
# ASE phase-1: engineai_locomotion 目录下全部 .npz，等权（同概率）采样。
#
# 用法（项目根目录）:
#   ./scripts/train_ase_smoke_engineai.sh
#   NUM_ENVS=2048 MAX_ITERS=3000 ./scripts/train_ase_smoke_engineai.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-smoke_engai1}"
AMP_DATA="${AMP_DATA:-src/assets/amp_motions/pnd_adam_sp/engineai_locomotion}"
VIDEO="${VIDEO:-True}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

# 默认：目录下全部 .npz 等权（同概率）；可用 AMP_DATASETS 覆盖
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

echo "[INFO] ASE smoke | envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
echo "[INFO] AMP data: $AMP_DATA"
echo "[INFO] AMP datasets (equal prob): $AMP_DATASETS"
echo "[INFO] Video: $VIDEO interval=$VIDEO_INTERVAL length=$VIDEO_LENGTH"

exec python scripts/train.py AdamSP-ASE-Flat \
  --env.scene.num-envs "$NUM_ENVS" \
  --amp-data-path "$AMP_DATA" \
  --amp-datasets "$AMP_DATASETS" \
  --video "$VIDEO" \
  --video-interval "$VIDEO_INTERVAL" \
  --video-length "$VIDEO_LENGTH" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.save-interval "$SAVE_INTERVAL" \
  --agent.run-name "$RUN_NAME"
