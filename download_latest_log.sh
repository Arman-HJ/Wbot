#!/bin/bash

set -u

# 用法:
#   ./download_latest_log.sh              -> 下载全部比本地最新 run 更新的目录（默认不限制数量）
#   ./download_latest_log.sh x2_amp       -> 仅指定实验
#   ./download_latest_log.sh x2_amp 500   -> 指定实验，仅 model_<id> 且 id % 500 == 0
#   ./download_latest_log.sh 500          -> 全部预设 + 模型间隔 500
#   MAX_RUNS_PER_EXPERIMENT=0 ./download_latest_log.sh  -> 下载全部比本地更新的 run（仍跳过更旧日期）

# 远程服务器配置
SERVER_USER="songhat"
SERVER_IP="192.168.31.249"
SERVER_PORT="22"
REPO_ROOT="/home/songhat/code_workspace/rl/unitree_rl_mjlab"

# 每个实验最多下载多少个「比本地更新」的 run（0=不限制数量，仍只比本地新）
MAX_RUNS_PER_EXPERIMENT="${MAX_RUNS_PER_EXPERIMENT:-0}"

# 默认要同步的实验（无参数时全部处理，不交互询问）
PRESET_KEYS=("adam_sp_velocity_v3" "adam_sp_amp_v2" "x2_standing_legs")
declare -A LOG_PRESETS=(
  [adam_sp_amp_v2]="logs/rsl_rl/adam_sp_amp_v2"
  [adam_sp_velocity_v3]="logs/rsl_rl/adam_sp_velocity_v3"
  [x2_standing_legs]="logs/rsl_rl/x2_standing_legs"
)

MODEL_INTERVAL=0
LOGS_KEYS=()

is_numeric() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

resolve_logs_path() {
  local key="$1"
  local path="${LOG_PRESETS[$key]:-}"
  if [ -z "${path}" ]; then
    echo "错误: 未知实验 \"${key}\"，可选: ${PRESET_KEYS[*]}" >&2
    return 1
  fi
  echo "${path}"
}

# 解析参数
if [ $# -ge 1 ]; then
  if is_numeric "$1"; then
    MODEL_INTERVAL="$1"
  else
    LOGS_KEYS=("$1")
    if [ $# -ge 2 ]; then
      if is_numeric "$2"; then
        MODEL_INTERVAL="$2"
      else
        echo "错误: 第二个参数应为模型编号间隔（非负整数）" >&2
        exit 1
      fi
    fi
  fi
fi

if ! is_numeric "${MODEL_INTERVAL}"; then
  echo "错误: 模型编号间隔必须是非负整数" >&2
  exit 1
fi

# 无参数：处理 PRESET_KEYS 中全部实验
if [ ${#LOGS_KEYS[@]} -eq 0 ]; then
  LOGS_KEYS=("${PRESET_KEYS[@]}")
fi

CURRENT_DIR=$(pwd)
RSYNC_SSH="ssh -p ${SERVER_PORT}"

sync_selected_models() {
  local logs_path="$1"
  local run_dir="$2"
  local interval="$3"
  local remote_run="${REPO_ROOT}/${logs_path}/${run_dir}"
  local local_run="${CURRENT_DIR}/${logs_path}/${run_dir}"

  if [ "${interval}" -le 0 ]; then
    return 0
  fi

  local remote_models
  remote_models=$(ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" \
    "cd '${remote_run}' && ls -1 model_*.pt 2>/dev/null" || true)

  if [ -z "${remote_models}" ]; then
    echo "  - 未发现 model_*.pt，跳过模型间隔筛选"
    return 0
  fi

  echo "  - 按间隔 ${interval} 筛选模型文件..."
  while IFS= read -r mf; do
    [ -z "${mf}" ] && continue
    local mid
    mid=$(echo "${mf}" | sed -n 's/^model_\([0-9]\+\)\.pt$/\1/p')
    [ -z "${mid}" ] && continue
    if [ $((mid % interval)) -eq 0 ]; then
      rsync -az --info=stats2 -e "${RSYNC_SSH}" \
        "${SERVER_USER}@${SERVER_IP}:${remote_run}/${mf}" \
        "${local_run}/"
    fi
  done <<< "${remote_models}"
}

latest_local_run_dir() {
  local local_logs_dir="$1"
  if [ ! -d "${local_logs_dir}" ]; then
    echo ""
    return 0
  fi
  ls -1 "${local_logs_dir}" 2>/dev/null \
    | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$' \
    | sort \
    | tail -n 1
}

download_missing_for_experiment() {
  local logs_key="$1"
  local logs_path
  logs_path=$(resolve_logs_path "${logs_key}") || return 1
  local local_logs_dir="${CURRENT_DIR}/${logs_path}"
  local remote_logs_abs="${REPO_ROOT}/${logs_path}"

  echo "=========================================="
  echo "实验: ${logs_key}"
  echo "远程路径: ${remote_logs_abs}"
  echo "本地路径: ${local_logs_dir}"
  echo "=========================================="

  mkdir -p "${local_logs_dir}"

  echo "正在查找远程 log 目录..."
  local remote_dirs
  remote_dirs=$(ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" \
    "cd '${remote_logs_abs}' && ls -d */ 2>/dev/null | sed 's/\///'" 2>/dev/null) || true

  if [ -z "${remote_dirs}" ]; then
    echo "警告: 远程无 run 目录或无法访问: ${remote_logs_abs}"
    echo ""
    return 0
  fi

  local latest_local
  latest_local=$(latest_local_run_dir "${local_logs_dir}")
  if [ -n "${latest_local}" ]; then
    echo "本地最新 run: ${latest_local}（仅下载比它更新的目录）"
  else
    echo "本地尚无 run 目录（将下载全部远程缺失 run；若只想拉最新可设 MAX_RUNS_PER_EXPERIMENT=1）"
  fi

  local newer_dirs=""
  for dir in ${remote_dirs}; do
    if [ -n "${latest_local}" ] && [[ "${dir}" < "${latest_local}" || "${dir}" == "${latest_local}" ]]; then
      continue
    fi
    if [ -d "${local_logs_dir}/${dir}" ]; then
      continue
    fi
    newer_dirs="${newer_dirs} ${dir}"
  done
  newer_dirs=$(echo "${newer_dirs}" | sed 's/^ *//')

  if [ -z "${newer_dirs}" ]; then
    if [ -n "${latest_local}" ]; then
      echo "没有比 ${latest_local} 更新的缺失 run，跳过 ${logs_key}"
    else
      echo "无可下载 run，跳过 ${logs_key}"
    fi
    echo ""
    return 0
  fi

  local newer_count
  newer_count=$(echo "${newer_dirs}" | wc -w | tr -d ' ')
  local dirs_to_download="${newer_dirs}"
  if [ "${MAX_RUNS_PER_EXPERIMENT}" -gt 0 ] && [ "${newer_count}" -gt "${MAX_RUNS_PER_EXPERIMENT}" ]; then
    dirs_to_download=$(echo "${newer_dirs}" | tr ' ' '\n' | grep -v '^$' | sort -r | head -n "${MAX_RUNS_PER_EXPERIMENT}" | tr '\n' ' ')
    dirs_to_download=$(echo "${dirs_to_download}" | sed 's/ *$//')
    local skipped=$((newer_count - MAX_RUNS_PER_EXPERIMENT))
    echo "远程有 ${newer_count} 个比本地新的缺失 run，跳过 ${skipped} 个（仍比本地新但非最新）"
    echo "仅下载最新 ${MAX_RUNS_PER_EXPERIMENT} 个: ${dirs_to_download}"
  else
    echo "自动下载: ${dirs_to_download}"
  fi

  for dir in ${dirs_to_download}; do
    echo "开始下载 ${logs_key}/${dir}..."
    mkdir -p "${local_logs_dir}/${dir}"

    if [ "${MODEL_INTERVAL}" -gt 0 ]; then
      rsync -az --info=stats2 -e "${RSYNC_SSH}" \
        --exclude="videos/" \
        --exclude="model_*.pt" \
        "${SERVER_USER}@${SERVER_IP}:${remote_logs_abs}/${dir}/" \
        "${local_logs_dir}/${dir}/"
      sync_selected_models "${logs_path}" "${dir}" "${MODEL_INTERVAL}"
    else
      rsync -az --info=stats2 -e "${RSYNC_SSH}" \
        --exclude="videos/" \
        "${SERVER_USER}@${SERVER_IP}:${remote_logs_abs}/${dir}/" \
        "${local_logs_dir}/${dir}/"
    fi

    if [ $? -eq 0 ]; then
      echo "${dir} 下载成功"
      du -sh "${local_logs_dir}/${dir}"
    else
      echo "错误: ${dir} 下载失败"
    fi
  done
  echo ""
}

echo "=========================================="
echo "从远程服务器下载最新 log"
echo "=========================================="
echo "服务器: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "模型编号间隔: ${MODEL_INTERVAL} (0=全部)"
echo "每实验最多下载比本地新的 run 数: ${MAX_RUNS_PER_EXPERIMENT} (0=全部比本地新)"
echo "过滤目录: videos/"
echo "待处理实验: ${LOGS_KEYS[*]}"
echo "=========================================="

for logs_key in "${LOGS_KEYS[@]}"; do
  download_missing_for_experiment "${logs_key}"
done

echo "=========================================="
echo "所有下载完成!"
echo "=========================================="
