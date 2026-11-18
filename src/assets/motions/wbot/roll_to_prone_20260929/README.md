# WBOT 仅翻身至趴稳

范围：仰躺 → 侧翻 → 对齐已训练起身动作的第0帧关节姿态 → 趴稳并保持。没有执行后续起身策略，没有修改已有训练权重。

1.6秒翻身、0.5秒平滑过渡至原起身参考的初始关节目标，再保持3秒稳定，共5.1秒。播放完成后继续用末端关节目标保持趴姿；按R重播，空格暂停，关窗退出。

```bash
bash src/assets/motions/wbot/roll_to_prone_20260929/play.sh
```

参考：`roban_wbot20_no_state_estimation/prone_1_robans221_1011_wbot20.npz` 第0帧。末端关节角RMSE约0.0291rad（1.67°），末端最大关节速度约0.000791rad/s。root平移和离地高度由实际接触决定，未强行拷贝参考root。完整末态在`handoff_state.npz`，可作为现有起身段的连接状态；这次没有验证后续策略接管。

`rollout_motion.npz`是50Hz真实状态（含初始帧），`physics_rollout.npz`是250Hz逐步控制/状态/力矩；`controller.npz`保存初态及全部关节目标；`simulation_model.xml/.mjb`是独立仿真模型；`analysis/`保存地面和关节检查。仅初始化一次，其余均为无外加root力的真实MuJoCo正向积分。

模型沿用前轮碰撞修正版：髋限位[-2.8,1.57]、零质量外壳地面网格、左右脚壳接触pair、脚底摩擦0.6、原执行器限制，步长0.004秒。未覆盖项目原机器人描述文件。

本段逐步重放1275步误差为0；最后1秒稳定趴姿比例100%，无超过5mm的腾空；最大关节速度为配置上限的80.9%，力矩未超上限。可视网格最大软接触入地2.20mm。通用审计中的站立比例0属于预期，本段任务判据见`analysis/task_check.json`。

已保存双视角视频：[roll_to_prone_replay.mp4](video/roll_to_prone_replay.mp4)，1280×720、50fps、2.62秒（趴稳后仅停留约0.5秒），按已验证的真实仿真状态渲染。

右脚修正版：翻身中段右脚相对骨盆的朝向偏转峰值由约71°降至约5°，踝侧翻峰值由约21°降至约5°。通过髋偏航补偿和减小踝侧翻目标实现，并完整重新进行动力学仿真。原版本保留于 `versions/before_right_foot_fix/`。MuJoCo软关节约束最大瞬时超界约0.00565rad，未检查完整三角网格自交。

训练参考 `tracking_motion.npz` 为修正版前131帧（50Hz，末帧2.60秒）。独立任务 `Wbot-20Dof-Tracking-RollToProne`，启动脚本 `scripts/train_tracking_roll_wbot_20dof.sh`，末帧保持趴姿，episode 3秒。训练模型保留同样的外壳接触和髋限位，未覆盖原起身任务。
