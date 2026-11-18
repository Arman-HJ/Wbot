# `motions/` — 动作模仿 / tracking 参考

给 `MotionCommand.motion_file` 用，**不进 AMP 判别器**。

| 路径 | 任务 |
|------|------|
| `wbot_20dof/gvhmr_kick/` | Kick（默认 `kick4_wbot_20dof.npz`；另有 `kick3_…stand2s.npz`；`kick5/6/7/8_wbot_20dof.npz` 由 GMR pkl 经 `scripts/motion_npz_to_tracking.py --fps 50 --clip-limits --ground-fix` 生成，kick5 未加 `--ground-fix`） |
| `wbot_12dof/gvhmr_kick/` | 12DoF Kick |
| `wbot_12dof/football_csv/`、`football_right_kick/` | football CSV 转 npz |
| `wbot_20dof/prone_*`、`Getting_up-*`、`prone_grounded_start/`、`prone_support_v4/` | 起立 tracking |
| `wbot/roban_wbot20_no_state_estimation/`、`wbot/roll_to_prone_20260929/` | 起立 / 翻身 |
| `x2/`、`g1/`、`g1_23dof/` | 其它机器人 tracking |
