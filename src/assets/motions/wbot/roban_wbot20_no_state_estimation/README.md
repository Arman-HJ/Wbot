用户提供的起身动作数据，原始位置：工作区根目录。
851 帧，50 FPS；原 771 帧版本保留在 ../roban_wbot20_step_getup。
训练任务：Wbot-20Dof-Tracking-Getup；actor 移除 motion_anchor_pos_b 和 base_lin_vel，critic 保留特权观测。
参考运行：logs/rsl_rl/wbot_20dof_getup_tracking/2026-09-24_17-37-44_wbot_20dof_roban_step_getup。
因 actor 输入维度变化，从头训练。
