#!/usr/bin/env bash
# 常用训练 / play / 数据准备命令备忘（复制单行执行，勿直接 bash cmd.sh）。
# 工作目录: cd /path/to/unitree_rl_mjlab
# 任务名为 positional 参数；后台: nohup ... < /dev/null > xxx.log 2>&1 &

# =============================================================================
# AgiBot X2 — Velocity v1 (legacy flat)  |  logs/rsl_rl/x2_velocity/
# =============================================================================

nohup python scripts/train.py AgiBot-X2-Flat --env.scene.num-envs=4096 < /dev/null > train_x2.log 2>&1 &

python scripts/play.py AgiBot-X2-Flat --run 2026-04-27_11-16-08/model_1600.pt

# =============================================================================
# Unitree G1 — Velocity v1 (legacy flat)  |  logs/rsl_rl/g1_flat/
# =============================================================================

nohup python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096 < /dev/null > train_g1.log 2>&1 &

python scripts/play.py Unitree-G1-Flat --run logs/rsl_rl/g1_flat/2026-05-06_17-31-22/model_1000.pt

# =============================================================================
# AgiBot X2 — Velocity v2 (motion-guided)  |  logs/rsl_rl/x2_velocity_v2/
# =============================================================================

python scripts/pkl_to_npz.py --input-file /home/songhat/code_workspace/Motion_retargeting/GMR/retargeted_datasets/agibot_x2/run/run4_agibot_x2.pkl --output-file src/assets/motions/x2/run4_agibot_x2.npz --robot agibot_x2_full --export-base-lin-vel-curve --base-body-name pelvis --curve-output-dir src/assets/motions/x2 --preview

nohup python scripts/train.py AgiBot-X2-Velocity-v2-Flat --motion-file=src/assets/motions/x2/run3_agibot_x2.npz --env.scene.num-envs=4096 < /dev/null > train_x2_velocity_v2_run3.log 2>&1 &

python scripts/play.py AgiBot-X2-Velocity-v2-Flat --run 2026-04-29_17-32-35/model_1000.pt --motion-file src/assets/motions/x2/run3_agibot_x2.npz

# =============================================================================
# AgiBot X2 — Velocity v3（全身 ~3 m/s，无需 motion npz）  |  logs/rsl_rl/x2_velocity_v3/
# =============================================================================

nohup python scripts/train.py AgiBot-X2-Velocity-v3-Flat --env.scene.num-envs=4096 < /dev/null > train_x2_velocity_v3.log 2>&1 &

# 接续: --checkpoint_file + --agent.max-iterations（总迭代上限）
# nohup python scripts/train.py AgiBot-X2-Velocity-v3-Flat-Enhanced --env.scene.num-envs=4096 --checkpoint_file=/path/to/model_100000.pt --agent.max-iterations 150000 < /dev/null > train_x2_velocity_v3_flat_enhanced.log 2>&1 &

python scripts/play.py AgiBot-X2-Velocity-v3-Flat --run 2026-05-27_21-14-55/model_43200.pt

# nohup python scripts/train.py AgiBot-X2-Velocity-v3-Rough --env.scene.num-envs=4096 < /dev/null > train_x2_velocity_v3_rough.log 2>&1 &

# =============================================================================
# AgiBot X2 — AMP v2（rsl_rl PPOAMP）  |  logs/rsl_rl/x2_amp_v2/
# =============================================================================

python3 scripts/train.py AgiBot-X2-AMP-v2-Flat \
  --env.scene.num-envs=4096 \
  --amp-data-path=src/assets/amp_motions/x2/engineai_locomotion \
  --amp-datasets "locomotion_agibot_x2:1.0,locomotion_agibot_x2_sym:1.0"

# =============================================================================
# AgiBot X2 — AMP v1（concat s/s'）  |  logs/rsl_rl/x2_amp/
# =============================================================================

python3 scripts/pkl_to_npz.py --input-dir /home/songhat/code_workspace/Motion_retargeting/GMR/retargeted_datasets/agibot_x2/robot_stand_walk_run --output-dir src/assets/amp_motions/x2/robot_stand_walk_run
# 仅检查: python3 -c "from scripts.pkl_to_npz import check_pkl_joint_order; check_pkl_joint_order('/home/songhat/code_workspace/Motion_retargeting/GMR/retargeted_datasets/agibot_x2/robot_stand_walk_run/stand-walk_agibot_x2.pkl')"
# 列序修复: python3 scripts/fix_npz_joint_order.py src/assets/amp_motions/x2/robot_stand_walk_run/*.npz --inplace --backup
# 训练前校验: python3 scripts/verify_amp_dataset.py --amp-dir src/assets/amp_motions/x2/robot_stand_walk_run
# 播 npz: python scripts/play_npz.py --npz-file src/assets/amp_motions/x2/robot_stand_walk_run/stand-walk_agibot_x2.npz

nohup python3 scripts/train.py AgiBot-X2-AMP-Flat --amp-data-path=src/assets/amp_motions/x2/robot_stand_walk_run --env.scene.num-envs=4096 --amp-datasets "stand-walk_agibot_x2:1.0,walk-run_agibot_x2:0,run-walk_agibot_x2:0,walk-stand_agibot_x2:0" < /dev/null > train_x2_amp_stand_walk_run.log 2>&1 &

nohup python3 scripts/train.py AgiBot-X2-AMP-Flat --amp-data-path=src/assets/amp_motions/x2/robot_stand_walk_run --env.scene.num-envs=4096 --amp-datasets "stand-walk_agibot_x2:0.1,walk-run_agibot_x2:1.0,run-walk_agibot_x2:1.0,walk-stand_agibot_x2:0.1" --checkpoint_file=logs/rsl_rl/x2_amp/2026-05-19_15-04-52/model_2300.pt --agent.max-iterations 100000 < /dev/null > train_x2_amp_stand_walk_run_resume.log 2>&1 &

python3 scripts/play.py AgiBot-X2-AMP-Flat --amp-data-path=src/assets/amp_motions/x2/robot_stand_walk_run --run logs/rsl_rl/x2_amp/2026-05-21_11-49-29/model_33300.pt


# =============================================================================
# Unitree H2 — Velocity Flat  |  logs/rsl_rl/h2_velocity/
# =============================================================================

# nohup python scripts/train.py Unitree-H2-Flat --env.scene.num-envs=4096 \
#   --agent.max-iterations 10001 --agent.save-interval 500 --agent.run-name h2_velocity \
#   < /dev/null > logs/rsl_rl/_console/train_h2.log 2>&1 &

python scripts/play.py Unitree-H2-Flat --viewer viser --num-envs=1 \
  --run 2026-09-16_15-50-17_h2_velocity/model_10000.pt

# =============================================================================
# WBOT 12DoF — Kick（tracking + ball）  |  logs/rsl_rl/wbot_12dof_kick/
# =============================================================================

# ./scripts/train_kick_wbot_12dof.sh
# 或: NUM_ENVS=4096 nohup ./scripts/train_kick_wbot_12dof.sh \
#   < /dev/null > logs/rsl_rl/_console/train_kick.log 2>&1 &

python scripts/play.py Wbot-12Dof-Kick --viewer viser --num-envs=1 \
  --run 2026-09-16_16-45-25_wbot_12dof_kick/model_30000.pt

# =============================================================================
# WBOT 12DoF — AMP v2 Flat  |  logs/rsl_rl/wbot_12dof_amp_v2/
# Viser：绑 0.0.0.0:8080；Mac 局域网打不开可加 --viser-share
# =============================================================================

# ./scripts/train_amp_wbot_12dof.sh
# 或: NUM_ENVS=4096 nohup ./scripts/train_amp_wbot_12dof.sh \
#   < /dev/null > logs/rsl_rl/_console/train_amp.log 2>&1 &

python scripts/play.py Wbot-12Dof-AMP-v2-Flat --viewer viser --num-envs=1 \
  --run 2026-09-16_14-28-16_wbot_12dof_amp_v2/model_9500.pt

# python scripts/play.py Wbot-12Dof-AMP-v2-Flat --viewer viser --viser-share --num-envs=1 \
#   --run 2026-09-16_14-28-16_wbot_12dof_amp_v2/model_9500.pt

# =============================================================================
# WBOT 20DoF — AMP v2 Flat  |  logs/rsl_rl/wbot_20dof_amp_v2/
# 数据: src/assets/amp_motions/wbot_20dof/engineai_locomotion
# =============================================================================

# ./scripts/train_amp_wbot_20dof.sh
# 或: NUM_ENVS=4096 nohup ./scripts/train_amp_wbot_20dof.sh \
#   < /dev/null > logs/rsl_rl/_console/train_amp_20dof.log 2>&1 &

# python scripts/play.py Wbot-20DoF-AMP-v2-Flat --viewer viser --num-envs=1 \
#   --run <run>/model_<iter>.pt

# WBOT 20DoF — Kick（WASM play）
# 只传 run 目录即可：网页下拉自动同步全部 model_*.pt（默认最新）；关闭: Ctrl+C 或 pkill -f 'vite.*--port 8080'
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Wbot-20Dof-Kick \
  --run 2026-11-02_16-06-05_kick8_15500_kickonly_ampstand_20261102_160602 \
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8080

# WBOT 20DoF — Victory2（kickcfg / WASM play）→ http://127.0.0.1:8080
# 只传 run 目录；网页下拉同步 model_*.pt（默认最新）；关闭: Ctrl+C 或 pkill -f 'vite.*--port 8080'
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Wbot-20Dof-Tracking-Victory2 \
  --run 2026-11-01_19-23-33_victory2_pace_20261101_192329\
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8080

# WBOT 20DoF — Getup 起身（roban / WASM play）→ http://127.0.0.1:8080
# 只传 run 目录；网页下拉同步 model_*.pt（默认最新）；motion 自动用 run 目录下的 .npz；关闭: Ctrl+C 或 pkill -f 'vite.*--port 8080'
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Wbot-20Dof-Tracking-Getup \
  --run 2026-10-25_23-44-09_getup_roban_start20 \
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8080

# =============================================================================
# WBOT LS-AMP walk/run/turn/side/stand  |  logs/rsl_rl/wbot_{12,20}dof_walk_run_amp/
# 数据: src/assets/amp_motions/wbot_20dof/walk_run_amp/  (预览 video/)
# --run 只写 run 名（自动在 logs/rsl_rl/<任务实验名>/ 下找）；--viewer wasm 网页同步全部 model_*.pt；关闭: Ctrl+C / pkill -f 'vite.*--port'
# =============================================================================

# 20DoF 训练（booster_recipe 设定 + 脚/ankle_roll mesh&collision −12mm；无 landing critic/soft_landing/foot_height DR）
# RUN_NAME=ankle12mm_geom_only_$(date +%Y%m%d_%H%M%S) NUM_ENVS=4096 MAX_ITERS=30000 \
#   nohup ./scripts/train_amp_walk_wbot_20dof.sh < /dev/null \
#   > logs/rsl_rl/_console/train_amp_ankle12mm_geom_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 12DoF（腿部）  → http://127.0.0.1:8081
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play_amp.py AMP-Walk-Run-Wbot-12DOF \
  --run 2026-10-09_17-21-02_wbot_12dof \
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8081

# 20DoF  → http://127.0.0.1:8082
# 脚 collision+mesh 一起 −12mm 已 bake 进 xmls/wbot_20dof.xml
# 原版备份: xmls_backup_stock_20261026/；play 勿再叠 --ankle-roll-forward
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play_amp.py AMP-Walk-Run-Wbot-20DOF \
  --run 2026-11-02_18-00-03_pace_new_resume800.pt_20261102_180000 \
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8082 \
  --sole-height 0.0 \
  --ankle-roll-forward 0.0

# =============================================================================
# WBOT Suspended（肩吊绳 + 正弦腿/臂参考）  |  logs/rsl_rl/wbot_{12,20}dof_suspended/
# wasm 暂不支持 Suspended，仍用 Viser；关闭: pkill -f 'play.py Wbot-.*-Suspended'
# =============================================================================

# 12DoF  → http://127.0.0.1:8083
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Wbot-12Dof-Suspended \
  --run 2026-10-08_11-36-53_suspended_shoulder_hang_v7/model_1499.pt \
  --num-envs 1 \
  --viewer viser \
  --viser-host 127.0.0.1 \
  --viser-port 8083

# 20DoF（arm_swing）  → http://127.0.0.1:8084
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Wbot-20Dof-Suspended \
  --run 2026-10-10_10-57-43_suspended_20dof_arm_swing/model_499.pt \
  --num-envs 1 \
  --viewer viser \
  --viser-host 127.0.0.1 \
  --viser-port 8084


cd /home/songhat/code_workspace/unitree_rl_mjlab && \
./scripts/play_walk_kick.sh \
  --walk-checkpoint /home/songhat/code_workspace/unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_walk_run_amp/2026-10-15_18-00-09_landing_force_resume_20261015_180001/model_7400.pt \
  --kick-checkpoint /home/songhat/code_workspace/unitree_rl_mjlab/logs/rsl_rl/wbot_20dof_kick/2026-10-15_15-39-34_kick3_amp_dr_20261015_153926/model_10500.pt




# H2 临时电机 play（wasm，velocity profile；网页勾 Enable 后滑条/WASDQE 调速，vx 上限随训练配置 3 m/s）
conda activate env_mjlab
cd /home/songhat/code_workspace/unitree_rl_mjlab && \
python scripts/play.py Unitree-H2-TmpMotor-Flat \
  --run 2026-10-22_15-43-07_h2_tmp_motor_vmax3_pose2_hr330_hy120 \
  --num-envs 1 \
  --viewer wasm \
  --viser-host 127.0.0.1 \
  --viser-port 8085