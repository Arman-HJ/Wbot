#!/usr/bin/env bash
# SMP smoke（兼容旧入口）—— 默认只跑阶段1先验；要跑 RL 需 RUN_POLICY=1。
#
# 阶段1（默认）:
#   ./scripts/train_smp_smoke_engineai.sh
# 阶段2:
#   RUN_POLICY=1 ./scripts/train_smp_smoke_engineai.sh
#   或: ./scripts/train_smp_policy_engineai.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_POLICY="${RUN_POLICY:-0}"

if [[ "$RUN_POLICY" == "1" ]]; then
  # 若先验已存在可 SKIP_PRIOR=1
  if [[ "${SKIP_PRIOR:-0}" != "1" ]]; then
    ./scripts/train_smp_prior_engineai.sh
  fi
  exec ./scripts/train_smp_policy_engineai.sh
fi

exec ./scripts/train_smp_prior_engineai.sh
