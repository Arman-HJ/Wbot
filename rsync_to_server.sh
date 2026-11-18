#!/bin/bash

# rsync 脚本：将当前代码推送到服务器
# 用法: ./rsync_to_server.sh [服务器地址]
# 示例: ./rsync_to_server.sh user@server:/path/to/destination

user=songhat
server_ip=192.168.31.249
project_name="unitree_rl_mjlab"
SOURCE_DIR="/home/songhat/code_workspace/rl/$project_name"
DESTINATION_DIR="/home/songhat/code_workspace/rl/$project_name"

# rsync 选项：
# -a: archive mode (递归、保持属性)
# -v: verbose
# -z: compress
# --delete: 删除目标中不存在的文件
# --exclude: 排除指定文件/目录

rsync -avz \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='.vscode/' \
    --exclude='wandb/' \
    --exclude='*.egg-info/' \
    --exclude='.git/' \
    --exclude='node_modules/' \
    --exclude='*.log' \
    --exclude='MUJOCO_LOG.TXT' \
    --exclude='WANDB_INTEGRATION.md' \
    --exclude='CORE_FLOW_ANALYSIS.md' \
    --exclude='download_latest_log.sh' \
    --exclude='pnd_gif/' \
    --exclude='img/' \
    --exclude='outputs/' \
    --exclude='source/whole_body_tracking/whole_body_tracking/assets' \
    --exclude='resources/robots/adam_lite/' \
    --exclude='resources/robots/adam_pro/' \
    --exclude='pre_train/' \
    --exclude='scripts/test_wandb.py' \
    --exclude='scripts/print_dofs.py' \
    --exclude='scripts/urdf2mjcf.py' \
    --exclude='deploy/deploy_mujoco/action_logs' \
    --exclude='src/assets/robots/agibot_x2/x2_assets/' \
    --exclude='src/assets/robots/adam_sp/meshes/z1_assets' \
    --ignore-times \
    "$SOURCE_DIR/" "$user@$server_ip:$DESTINATION_DIR"

echo "代码同步完成。"
