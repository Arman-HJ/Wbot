# `amp_motions/` — AMP 判别器专家数据

只放进 `amp_motion_files` / `--amp-data-path` 的包。纯 tracking / Kick 模仿在 `../motions/`。

| 路径 | 任务 |
|------|------|
| `wbot_20dof/walk_run_amp/` | LS-AMP 走跑转侧站（主用） |
| `wbot_12dof/walk_run_amp/` | 同上，12DoF |
| `wbot_20dof/getup/` | Get-up AMP |
| `wbot_20dof/recovery/` | 旧 getup AMP |
| `wbot_{12,20}dof/engineai_locomotion/` | 旧 / amp_v2 |
| `x2/**`、`pnd_adam_sp/**` | 其它机器人 AMP |

`walk_run_amp/turn_*_foot_height_profile.npz` 是 AMP 走路奖励脚高曲线，不是 tracking 参考。
