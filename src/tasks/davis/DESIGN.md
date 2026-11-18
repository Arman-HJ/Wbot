# DAVIS → wbot 设计说明（定向射门）

依据：[arXiv:2609.28175](https://arxiv.org/abs/2609.28175)、[项目页](https://thusi-lab.github.io/DAVIS/)。官方仓库无训练代码，本实现严格按论文公开内容迁移。

## 1. 仿真平台差异

| 项 | 论文 | 本实现 |
| --- | --- | --- |
| 仿真 | Isaac Sim 5.0.0 + Isaac Lab | **mjlab 1.5.3 + MuJoCo-Warp** |
| 深度渲染 | Isaac `distance_to_image_plane` | `CameraSensor` depth |
| 并行 | 论文附录：1024 envs（射门） | 同默认；sanity 用更少 |
| 资产 | Noetix E1 USD | wbot MJCF + **2-DoF 脖子嫁接** |

算法接口（深度→残差动作、特权仅给 critic/aux）与论文一致；物理/渲染栈不同，文档中不假装等价于 Isaac Lab。

## 2. 与 E1 的机器人差异

| 项 | Noetix E1 | wbot（DAVIS 任务） |
| --- | --- | --- |
| 驱动 DoF | 23 身躯 + 2 头 = 25 | **20 身躯 + 2 脖 = 22** |
| 主动头部 | 原生 yaw/pitch | **嫁接** `neck_yaw` + `neck_pitch` → `head_camera_link` |
| 本体感受单帧 | 射门 81-D（3+3+25+25+25） | **72-D**（3+3+22+22+22） |
| 动作 | `a ∈ R^25`，`q_des = q0 + s ⊙ a` | `a ∈ R^22`，同一公式 |

按 [项目页 #app-sim2real](https://thusi-lab.github.io/DAVIS/#app-sim2real) / 论文：头在感知闭环里；ablation 锁头后射门几乎崩溃 → 本实现为 wbot **加上脖子 DoF**，恢复 head-track / body-follow-head / head-yaw-align。

## 3. 相机（ZED2i，头戴）

对齐项目页：校准 ZED2i `distance_to_image_plane`，168×94→168×80。

| 参数 | 值 | 备注 |
| --- | --- | --- |
| 型号 | StereoLabs ZED2i（仿真） | 真机若仍用 D455，只换 FOV/外参，保持同一 168×80 接口 |
| Depth FOV | ≈92° H × 60° V | 整流后近似；`fovy=60°` |
| 仿真渲染 | 168×94 | 与论文一致 |
| 策略输入 | 裁切 **168×80** | 与论文一致 |
| 深度范围 / 归一化 | 0.3–5.0 m，`2D/5−1`，无效填 0 | Appendix F / 项目页 |
| 安装 | `head_camera_link` | 主动头带动下一帧深度 |
| 光轴 | 头坐标系 +X | MuJoCo quat `(0.5,0.5,-0.5,-0.5)` |
| 位姿 DR | 平移 ±0.03 m，姿态 ±0.04 rad | 项目页 0.02–0.04 m / 0.025–0.060 rad |
| 深度流率 | 15 Hz（部署） | 仿真按策略步渲染 |

配置：`configs/davis/camera_zed2i.yaml`。

## 4. 观测 / 动作

### Actor（部署路径，禁止球特权 XYZ）

1. 单帧深度 `168×80`（预处理后）  
2. 5 步本体感受历史（每步 72-D）：基座角速度、投影重力、相对关节位置/速度、上一步动作  
3. 射门任务：**无操作员指令**  
4. 训练时由深度 latent 经 aux 头得到的 **visibility-gated 几何特征**（部署时 β=1，纯预测）

### Critic / 奖励 / aux 监督（仅训练）

球/门在 **head-yaw 对齐系**下的中心、可见性（完整头姿 + FOV/深度）、基座线速度、足端接触、课程变量等。

### 动作

22 维关节位置残差（含 2 脖）→ PD：`q_des = q0 + s ⊙ a`。  
物理 200 Hz、策略 50 Hz、深度约 15 Hz。

## 5. 网络（论文给定部分 + 标注假设）

| 模块 | 论文 | 本实现 |
| --- | --- | --- |
| LightDepthEncoder → 32-D | 有名字与输出维，**无层结构** | 轻量 CNN + Linear→32，注释标明假设 |
| Aux geo / vis heads | 球+门各 3-D + logit | Linear 小头，结构未给出处标假设 |
| HIM | 本体历史 → **3-D** 隐状态估计 | MLP→3，结构未给出处标假设 |
| Actor / Critic MLP | `[512,256,128]` ELU | 同论文 |
| GT→pred annealing | √ ramp，可按 aux EMA 暂停 | 同公式；`k0/k1`/EMA 阈值为可配置（论文未给数值） |

Actor 特征拼接（概念）：`[z_depth(32), gated_aux(8), proprio_hist, him(3)]`。

## 6. 损失与优化

- PPO：rollout 24、5 epoch、4 mini-batch、lr=1e-3 adaptive、γ=0.99、λ=0.95、clip=0.2、entropy=0.01（Table X）  
- Aux：`λ_geo(ball/goal)=1.0`，`λ_vis=0.2`；几何仅在可见时监督（Eq. 3 / App. D-B）  
- AMP：论文 coef 0.8；**默认开启**（`DavisAmpPPO`）。动作包：HumanoidSoccer `wbot-soccer-standard` → `src/assets/amp_motions/wbot_20dof/soccer_standard/`（20-DoF，头/脖不进 disc）。加速对齐 football_jax：`num_envs=4096`、horizon 24、grad_pen 5、接触预算 nconmax/njmax 收紧。

## 7. 射门任务与课程

- 物体集：`{ball, goal}`；奖励权重按 Table XI  
- 主动视觉项：`head_track_ball` / `body_follow_head` / `head_yaw_align` / `keep_ball_in_horizon`  
- 初始化：penalty（机器人在球后 2–3.5 m、±100°）与 free-kick 区域按论文；课程逐步放宽初始分布、逐步打开对准/击球项（阶段阈值可配置）  
- 成功：球一次击球后穿过球门线

## 8. 目录

```
src/tasks/davis/
  DESIGN.md
  mdp/          # 观测、奖励、事件、终止、课程、深度预处理
  rl/           # LightDepthEncoder、DavisActor、DavisPPO
  config/wbot_20dof/
configs/davis/  # camera / sanity / full YAML
scripts/train_davis_*.sh, play_davis_*.sh
```

## 9. 对齐清单

### 已严格按论文实现

- 部署接口：单帧深度 + 本体感受历史（+ 射门无指令）→ 关节残差；actor 前向不吃球特权 XYZ
- 深度：168×94 渲染 → 裁 168×80；0.3–5.0 m；`2D/5−1`；无效填 0
- 深度 latent 32-D；aux 球/门几何 + 可见性；visibility gate `clip(stopgrad(p),0.05,1)`
- GT→pred √ annealing（可按 aux EMA 暂停）
- Actor/Critic MLP `[512,256,128]` ELU；PPO/AMP 表内超参（AMP 默认关）
- 射门奖励权重表（Table XI）；1024 envs / 20 s / dt=0.005 / decimation=4
- 域随机化范围按 Table XIII（深度噪声/掩膜/质量/推力等子集）

### 因论文未公开而做的假设（均已注释 / 可配置）

- LightDepthEncoder / aux 头 / HIM 的具体层结构
- Annealing `k0/k1`、EMA 阈值；课程阶段 step 阈值
- 各奖励项的具体函数形状（论文只给名称与权重）
- 脖子关节限位 / PD（论文未公开 E1 头电机参数）→ 用小臂级电机假设
- LightDepthEncoder 等层结构；annealing / 课程数值
- 仿真后端：mjlab 而非 Isaac Lab；可见性用头姿+FOV 近似完整相机投影

### 上真机还缺什么

1. 真机头/云台或固定 D455 的最终外参与内参（仿真默认 ZED2i FOV）
2. 真机深度 15 Hz 与策略 50 Hz / PD 200 Hz 时间同步
3. ONNX 导出与部署运行时（本阶段未写 ROS）
4. AMP 射门参考动作包（可选但论文用了）
5. 草地 / 球摩擦等真机域随机与安全兜底

### 命令

```bash
# Sanity（64 envs / 200 iters）
bash scripts/train_davis_shoot_wbot_20dof.sh --sanity

# Full
bash scripts/train_davis_shoot_wbot_20dof.sh

# Play / 录视频
bash scripts/play_davis_shoot_wbot_20dof.sh --run <run> --video --video-length 300
```
