#!/bin/bash
# 将单次训练 run 中的部署文件同步到机器人 deploy_x2_real_5/checkpoints/<实验名>/<run>/。
# 实验名与 logs/rsl_rl 一致（x2_amp、x2_velocity_v3 等），run 子目录为时间戳。
# 仅传输：policy.onnx、若存在的 policy.onnx.data、params/deployment_alignment.yaml；
# 若 run 下有 motions/（FALCON 上肢测试包）一并同步。
#
# 用法:
#   ./rsync_to_robot.sh
#     无参：在 DEFAULT_EXPERIMENT 下选最新可部署 run
#   ./rsync_to_robot.sh x2_velocity_v3
#     临时改用其它实验名（在 logs/rsl_rl/<实验名>/ 下找最新 run）
#   ./rsync_to_robot.sh logs/rsl_rl/x2_amp/2026-05-19_17-04-34
#     同步指定 run 目录
#   RSYNC_EXPERIMENT=x2_amp ./rsync_to_robot.sh
#     环境变量覆盖 DEFAULT_EXPERIMENT（与第 1 参数二选一即可）

set -euo pipefail

SOURCE_DIR="/home/songhat/code_workspace/rl/unitree_rl_mjlab"

# =============================================================================
# 1) 真机目标（取消注释一组即可）
# =============================================================================
# --- AgiBot X2 ---
user=agi
server_ip=192.168.31.35
password=1
REMOTE_CHECKPOINTS_ROOT="/home/agi/songhat_workspace/deploy_real/checkpoints"
DEFAULT_EXPERIMENT="x2_amp_v2"

# --- PND humanoid ---
# user=pnd-humanoid
# server_ip=192.168.31.81
# password='pndxyz'
# REMOTE_CHECKPOINTS_ROOT="/home/pnd-humanoid/Documents/songhat_workspace/deploy_real/checkpoints"
# DEFAULT_EXPERIMENT="adam_sp_standing"

: "${password:?请在脚本顶部设置 password=}"

# 远端布局: <REMOTE_CHECKPOINTS_ROOT>/<实验名>/<run 时间戳>/
# 例: .../checkpoints/adam_sp_amp_v2/2026-07-30_14-00-00/
# 常用实验: adam_sp_amp_v2 | adam_sp_standing | x2_standing_legs | x2_velocity_v3 | x2_amp

remote_configs_dir_for_experiment() {
  local exp="$1"
  echo "${REMOTE_CHECKPOINTS_ROOT}/${exp}"
}

resolve_experiment() {
  if [[ -n "${RSYNC_EXPERIMENT:-}" ]]; then
    echo "$RSYNC_EXPERIMENT"
  elif [[ -n "${EXPERIMENT_OVERRIDE:-}" ]]; then
    echo "$EXPERIMENT_OVERRIDE"
  else
    echo "$DEFAULT_EXPERIMENT"
  fi
}

is_experiment_name() {
  local name="$1"
  [[ -d "$SOURCE_DIR/logs/rsl_rl/$name" ]]
}

find_latest_run_dir() {
  local experiment
  experiment="$(resolve_experiment)"
  local log_root="$SOURCE_DIR/logs/rsl_rl"
  local pattern="$log_root/$experiment/*"
  if [[ ! -d "$log_root/$experiment" ]]; then
    echo "错误: 实验目录不存在: $log_root/$experiment" >&2
    echo "  请修改脚本顶部 DEFAULT_EXPERIMENT，或传入存在的实验名" >&2
    return 1
  fi
  local d
  while IFS= read -r d; do
    d="${d%/}"
    if [[ -f "$d/policy.onnx" && -f "$d/params/deployment_alignment.yaml" ]]; then
      echo "$d"
      return 0
    fi
  done < <(ls -1dt $pattern/ 2>/dev/null || true)
  echo "错误: 在 $log_root/$experiment 下未找到含 policy.onnx 的 run" >&2
  return 1
}

usage() {
  echo "用法: $0 [实验名 | run 目录]" >&2
  echo "  无参: 在 DEFAULT_EXPERIMENT（当前=${DEFAULT_EXPERIMENT}）下选最新可部署 run" >&2
  echo "  实验名: 如 x2_amp、x2_velocity_v3（须在 logs/rsl_rl/<名>/ 存在）" >&2
  echo "  run 目录: 绝对路径或相对 $SOURCE_DIR" >&2
  echo "  环境变量 RSYNC_EXPERIMENT 可覆盖默认实验名" >&2
  exit 1
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

RUN_ARG=""
EXPERIMENT_OVERRIDE=""

if [[ -n "${1:-}" ]]; then
  if [[ -d "$1" ]] || [[ -d "$SOURCE_DIR/$1" ]] || [[ "$1" == */* ]]; then
    RUN_ARG="$1"
  elif is_experiment_name "$1"; then
    EXPERIMENT_OVERRIDE="$1"
  else
    echo "错误: 无法识别参数: $1（不是目录，也不是 logs/rsl_rl 下的实验名）" >&2
    exit 1
  fi
fi

ACTIVE_EXPERIMENT="$(resolve_experiment)"

if [[ -z "$RUN_ARG" ]]; then
  RUN_ARG="$(find_latest_run_dir || true)"
  if [[ -z "$RUN_ARG" ]]; then
    exit 1
  fi
  echo "[INFO] 实验: ${ACTIVE_EXPERIMENT}"
  echo "[INFO] 未指定 run，使用最新可部署: $(basename "$RUN_ARG")"
fi

if [[ -d "$RUN_ARG" ]]; then
  RUN_DIR="$(cd "$RUN_ARG" && pwd)"
elif [[ -d "$SOURCE_DIR/$RUN_ARG" ]]; then
  RUN_DIR="$(cd "$SOURCE_DIR/$RUN_ARG" && pwd)"
else
  echo "错误: 找不到目录: $RUN_ARG（也试过 \$SOURCE_DIR/$RUN_ARG）" >&2
  exit 1
fi

case "$RUN_DIR" in
"$SOURCE_DIR"/*) ;;
*)
  echo "错误: run 目录必须在仓库内: $SOURCE_DIR" >&2
  exit 1
  ;;
esac

# 从 run 路径推断实验名（决定本地 logs 子目录与远端 checkpoints 子目录）
RUN_EXPERIMENT="$(basename "$(dirname "$RUN_DIR")")"
REMOTE_CONFIGS_DIR="$(remote_configs_dir_for_experiment "$RUN_EXPERIMENT")"

if [[ -n "${ACTIVE_EXPERIMENT:-}" && "$ACTIVE_EXPERIMENT" != "$RUN_EXPERIMENT" ]]; then
  echo "[WARN] 选定实验为 ${ACTIVE_EXPERIMENT}，但 run 路径属于 ${RUN_EXPERIMENT}，远端将写入 ${RUN_EXPERIMENT} 目录"
fi

RUN_BASENAME="$(basename "$RUN_DIR")"
REMOTE_RUN_DIR="$REMOTE_CONFIGS_DIR/$RUN_BASENAME"
REMOTE_DEST="$user@$server_ip:$REMOTE_RUN_DIR"

[[ -f "$RUN_DIR/policy.onnx" ]] || {
  echo "错误: 缺少 $RUN_DIR/policy.onnx（请先 play 导出或检查路径）" >&2
  exit 1
}
[[ -f "$RUN_DIR/params/deployment_alignment.yaml" ]] || {
  echo "错误: 缺少 $RUN_DIR/params/deployment_alignment.yaml" >&2
  exit 1
}

echo "=========================================="
echo "同步到真机"
echo "  实验: ${RUN_EXPERIMENT}"
echo "  本地: $RUN_DIR"
echo "  远端根: ${REMOTE_CONFIGS_DIR}"
echo "  远端 run: $REMOTE_RUN_DIR"
echo "=========================================="

# 自动填 SSH 密码（sshpass 优先，否则用 SSH_ASKPASS）
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no)
if command -v sshpass >/dev/null 2>&1; then
  export SSHPASS="${password}"
  remote_ssh() { sshpass -e ssh "${SSH_OPTS[@]}" "$@"; }
  RSYNC_RSH=(sshpass -e ssh "${SSH_OPTS[@]}")
else
  ASKPASS="$(mktemp)"
  printf '%s\n' '#!/bin/sh' "printf '%s\\n' $(printf '%q' "$password")" >"$ASKPASS"
  chmod 700 "$ASKPASS"
  trap 'rm -f "$ASKPASS"' EXIT
  export SSH_ASKPASS="$ASKPASS"
  export SSH_ASKPASS_REQUIRE=force
  export DISPLAY="${DISPLAY:-:0}"
  remote_ssh() { setsid -w ssh "${SSH_OPTS[@]}" "$@"; }
  RSYNC_RSH=(setsid -w ssh "${SSH_OPTS[@]}")
fi

remote_ssh "$user@$server_ip" "mkdir -p $(printf '%q' "$REMOTE_RUN_DIR/params")"

rsync -avz -e "${RSYNC_RSH[*]}" "$RUN_DIR/policy.onnx" "$REMOTE_DEST/"
if [[ -f "$RUN_DIR/policy.onnx.data" ]]; then
  rsync -avz -e "${RSYNC_RSH[*]}" "$RUN_DIR/policy.onnx.data" "$REMOTE_DEST/"
fi
rsync -avz -e "${RSYNC_RSH[*]}" "$RUN_DIR/params/deployment_alignment.yaml" "$REMOTE_DEST/params/"
# Optional FALCON upper-body motion pack for real-robot testing
if [[ -d "$RUN_DIR/motions" ]]; then
  remote_ssh "$user@$server_ip" "mkdir -p $(printf '%q' "$REMOTE_RUN_DIR/motions")"
  rsync -avz -e "${RSYNC_RSH[*]}" "$RUN_DIR/motions/" "$REMOTE_DEST/motions/"
fi

echo "本地 run: $RUN_DIR"
echo "已同步到: $REMOTE_RUN_DIR"
echo "  - policy.onnx"
[[ -f "$RUN_DIR/policy.onnx.data" ]] && echo "  - policy.onnx.data"
echo "  - params/deployment_alignment.yaml"
[[ -d "$RUN_DIR/motions" ]] && echo "  - motions/ (upper14 npy playlist)"
echo "真机 YAML 可写为:"
echo "  policy_path: \"$REMOTE_RUN_DIR/policy.onnx\""
echo "  deployment_alignment: \"$REMOTE_RUN_DIR/params/deployment_alignment.yaml\""
[[ -d "$RUN_DIR/motions" ]] && echo "  motions_dir: \"$REMOTE_RUN_DIR/motions\""
