# Getup tracking 参考动作

`prone_1_robans221_1011_wbot20.npz`：851 帧，50 FPS（约 17 s）。

## 训练任务
`Wbot-20Dof-Tracking-Getup`（`src/tasks/tracking/config/wbot_20dof_roban_20260928`）

- Actor（无状态估计）：去掉 `motion_anchor_pos_b`、`base_lin_vel`，含 `projected_gravity`
- Critic（特权）：保留 body 相对项 + 世界系 EE（脚/肘）+ `projected_gravity`
- 碰撞：`collision="getup"`（身体可贴地）

```bash
./scripts/train_tracking_roban_getup_wbot_20dof.sh
```
