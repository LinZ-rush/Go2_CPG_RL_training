# 当前系统设计文档

## 1. 系统概述

当前系统在原始单层 CPG-RL 控制的基础上，引入了**双层层级控制架构**，实现了**半自主运动控制**：

- **步态自动切换**: High-Level 网络根据速度自动选择 Walk / Trot / Bound 步态
- **智能跳跃辅助**: 操作员请求跳跃 + 网络可行性检查 → 允许/拒绝
- **自适应跳跃高度**: 网络根据运动状态输出最优跳跃高度 [0.3, 0.6]m

---

## 2. 双层层级架构

```
┌─────────────────────────────────────────────────────┐
│                   High-Level Policy                  │
│  35维观测 → MLP [512,256,128] → 3维输出 → Sigmoid   │
│                     (5Hz)                            │
└────────────────────┬────────────────────────────────┘
                     │ 3维决策 (步态/可行性/高度)
                     ▼
              ┌──────────────┐
              │   适配层      │
              │  Sigmoid→物理 │
              └──────┬───────┘
                     │ 步态ID, 跳跃信号, 跳跃高度
                     ▼
┌─────────────────────────────────────────────────────┐
│                   Low-Level Policy                   │
│  68维观测 → MLP (预训练,冻结) → 16维CPG偏移          │
│                    (100Hz)                            │
└────────────────────┬────────────────────────────────┘
                     │ 16维 CPG 参数
                     ▼
┌─────────────────────────────────────────────────────┐
│               CPG + 逆运动学 (IK)                    │
│  CPG振荡器 + RL偏移 → 足端轨迹 → 12维关节目标位置    │
│                    (1000Hz)                           │
└─────────────────────────────────────────────────────┘
```

### 2.1 控制频率对齐

| 层级 | 频率 | decimation | 功能 |
|------|------|-----------|------|
| 物理仿真/定时器 | 1000Hz | 1 | CPG更新 + IK计算 |
| Low-Level Policy | 100Hz | 10 | CPG参数偏移推理 |
| High-Level Policy | 5Hz | 200 | 步态/跳跃决策 |

### 2.2 High-Level Policy (大脑, 5Hz)

**输入观测 (35维)**:
```
[0:3]   base_lin_vel * 2.0          # 基座线速度 (仿真从/mujoco/vel获取)
[3:6]   base_ang_vel * 0.25         # 基座角速度
[6:9]   projected_gravity           # 重力投影向量
[9:12]  commands * cmd_scale        # 速度指令 (vx, vy, ωz)
[12:15] last_high_actions           # 上一步High-Level输出 (sigmoid后)
[15:19] contact                     # 4脚接触状态 (bool→float)
[19:23] cpg_r                       # CPG径向状态
[23:27] cpg_theta                   # CPG相位状态
[27:31] cpg_r_dot                   # CPG径向速度
[31:35] cpg_theta_dot               # CPG相位速度
```

**输出 (3维, 经 Sigmoid 映射到 [0,1])**:

| 维度 | 含义 | 映射规则 | 输出范围 |
|------|------|---------|---------|
| [0] | 步态选择 | `floor(x * 3.99)` → {0,1,2,3} | 0=TROT, 1=WALK, 2=BOUND, 3=PRONK |
| [1] | 跳跃可行性 | 直接使用 [0,1] | >0.7 表示可跳 |
| [2] | 跳跃高度 | `x * 0.3 + 0.3` | [0.3, 0.6]m |

### 2.3 适配层 (Adapter)

适配层将 High-Level 的 3 维 sigmoid 输出转换为 Low-Level 可理解的物理量：

```python
# 步态ID: 均匀映射到 {0, 1, 2, 3}
gait_id = floor(sigmoid_output[0] * 3.99)

# 跳跃可行性评分
feasibility = sigmoid_output[1]  # [0, 1]

# 跳跃高度: 线性映射
height = sigmoid_output[2] * 0.3 + 0.3  # [0.3, 0.6]m
```

### 2.4 跳跃决策融合

跳跃不是完全自主的，而是**人机协作**：

```
跳跃信号 = 操作员按键请求 AND 网络可行性评分 > 0.7 AND 脚接触地面
```

训练时通过 10% 概率随机请求模拟操作员按键，网络学习在合适条件下输出高可行性评分。

### 2.5 Low-Level Policy (脊髓, 100Hz)

预训练并冻结的 Low-Level 策略，接收适配层输出的步态ID、跳跃信号、跳跃高度，结合传感器数据输出 16 维 CPG 参数偏移。

**输入观测 (68维)**:
```
[0:3]   base_ang_vel * 0.25         # 角速度
[3:6]   projected_gravity           # 重力投影
[6:9]   commands * cmd_scale        # 速度指令
[9]     jump_height                 # 适配层: 自适应跳跃高度
[10]    gait_id                     # 适配层: 网络选择的步态
[11]    jump_signal                 # 融合后的跳跃信号
[12:24] (dof_pos - default) * 1.0   # 关节位置偏差
[24:36] dof_vel * 0.05              # 关节速度
[36:52] last_low_actions            # 上一步Low-Level动作
[52:68] CPG state (r, θ, ṙ, θ̇)    # 4×4 CPG振荡器状态
```

---

## 3. 训练配置

### 3.1 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `tracking_vel` | 20.0 | 速度追踪 (最高优先级) |
| `tracking_command_phase` | 15.0 | 相位追踪 |
| `gait_velocity_matching` | 10.0 | 步态-速度匹配 (新增) |
| `jump_feasibility_correct` | 8.0 | 跳跃可行性判断准确性 (新增) |
| `smooth_gait_transition` | -2.0 | 惩罚频繁步态切换 (新增) |
| `orientation_roll` | -200 | 防侧翻 |
| `orientation_yaw` | -300 | 防偏航 |
| `collision` | -1 | 碰撞惩罚 |
| `action_rate` | -0.01 | 动作平滑 |
| `energy` | -0.001 | 能耗惩罚 |

### 3.2 步态-速度映射规则

训练中 `_reward_gait_velocity_matching()` 定义的最优步态：

| 速度范围 | 最优步态 | 过渡区域 |
|---------|---------|---------|
| < 0.5 m/s | Walk (ID=1) | 0.4~0.6 容忍 Walk/Trot |
| 0.5~1.5 m/s | Trot (ID=0) | 1.4~1.6 容忍 Trot/Bound |
| >= 1.5 m/s | Bound (ID=2) | — |

过渡区域使用滞后设计，错误步态只扣一半分，防止在边界频繁切换。

### 3.3 跳跃可行性判断 (5 个条件)

`_reward_jump_feasibility_correct()` 计算真实可行性标签：

1. **速度追踪良好**: `exp(-vel_error²/σ) > 0.85`
2. **步态相位准确**: CPG 相位误差 < 5%
3. **姿态稳定**: 重力投影水平分量² < 0.1
4. **脚接触地面**: 至少一只脚触地
5. **模式稳定**: 当前模式已保持 > 100 步 (0.5s)

网络学习预测可行性评分，预测误差越小奖励越高。

### 3.4 训练超参数

| 参数 | 值 |
|------|-----|
| 环境数 | 2048 |
| num_steps_per_env | 24 |
| 最大迭代 | 3000 |
| Episode 长度 | 20s |
| 学习率 | 1e-3 (自适应) |
| PPO clip | 0.2 |
| entropy_coef | 0.005 |
| 网络结构 | MLP [512, 256, 128] |
| init_noise_std | 0.3 |
| 速度指令范围 | [0, 2.0] m/s (课程扩展至 max 3.0) |

---

## 4. 训练结果分析 (3000 轮)

### 4.1 最终指标 (wandb-summary, iteration 3000)

| 指标 | 值 | 说明 |
|------|-----|------|
| `mean_reward` | 695.8 | 总平均奖励 |
| `mean_episode_length` | 100.14 步 | 平均 Episode 长度 |
| `rew_tracking_vel` | 12.48 | 速度追踪 (权重20, 最大~20) — 追踪质量 62.4% |
| `rew_tracking_command_phase` | 13.08 | 相位追踪 (权重15) — 表现优秀 87.2% |
| `rew_gait_velocity_matching` | 2.40 | 步态匹配 (权重10) — 仅 24%, 未充分收敛 |
| `rew_jump_feasibility_correct` | 7.46 | 可行性判断 (权重8) — 93.3%, 但存在退化问题(见4.4) |
| `rew_smooth_gait_transition` | -0.08 | 步态切换惩罚 — 非常低,切换不频繁 |
| `rew_orientation_roll` | -0.17 | 侧翻惩罚 — 姿态稳定 |
| `rew_orientation_yaw` | -0.02 | 偏航惩罚 — 很低 |
| `rew_energy` | -0.13 | 能耗 — 正常范围 |
| `rew_action_rate` | -0.03 | 动作平滑 — 很好 |
| `max_command_x` | 2.5 m/s | 速度课程已扩展到 2.5 |
| `mean_noise_std` | 0.77 | 探索噪声 |
| `learning_rate` | 1e-5 | 已自适应降到很低 |

### 4.2 关键数据阶段和突变分析

以下基于 wandb 实际数据 (run: yufufomp) 逐阶段分析训练过程中的重要突变及其原因。

#### 阶段 1: 快速收敛期 (Step 0~100)

| Step | tracking_vel | command_phase | gait_match | feasibility | noise_std |
|------|-------------|--------------|------------|-------------|-----------|
| 0 | 0.98 | 0.95 | 0.36 | 0.09 | 0.30 |
| 4 | 12.72 | 6.79 | 1.52 | 1.59 | 0.30 |
| 50 | 13.78 | 6.65 | 1.30 | 1.42 | 0.32 |
| 99 | 14.02 | 6.46 | 1.20 | 1.26 | 0.33 |

前 5 步 `tracking_vel` 从 0.98 猛涨到 12.72（已接近峰值）。原因：Low-Level 是预训练冻结的，机器人本来就会走路，High-Level 只需学会"不添乱"（输出接近中间值的步态ID），速度追踪就自然很高。

此阶段 `gait_match` 和 `feasibility` 都很低，网络尚未开始学习这些新任务。

#### 阶段 2: 第一次突变 — 探索代价 (Step 100~200)

| Step | tracking_vel | command_phase | gait_match | noise_std |
|------|-------------|--------------|------------|-----------|
| 99 | **14.02** | 6.46 | 1.20 | 0.33 |
| 199 | **7.92** | 7.25 | **2.34** | 0.36 |

**`tracking_vel` 从 14 暴跌到 8** (下降 43%)，同时 `gait_match` 从 1.2 上升到 2.3。

**原因**: 网络开始认真探索步态切换。前 100 步它"偷懒"只输出固定步态（追踪最高但步态匹配低）。当噪声探索发现步态切换能获得奖励后，网络开始尝试切换步态，但切换过程中步态不匹配导致速度追踪暂时恶化。这是**必要的探索代价**——要学新技能，就要暂时牺牲旧技能的表现。

#### 阶段 3: 步态学习黄金期 (Step 200~1000)

| Step | tracking_vel | gait_match | feasibility | noise_std |
|------|-------------|------------|-------------|-----------|
| 200 | 8.17 | 2.39 | 1.28 | 0.36 |
| 400 | 10.08 | 2.42 | **3.04** | 0.45 |
| 600 | 12.89 | **4.79** | **5.41** | 0.57 |
| 800 | 13.46 | **7.44** | **7.12** | 0.68 |
| 1000 | 13.69 | **7.47** | **7.42** | 0.75 |

这是训练中最健康的阶段，所有指标同步提升：
- `gait_match`: 2.4 → 7.5（3倍），步态选择越来越准确
- `feasibility`: 1.3 → 7.4（5.7倍），可行性判断快速收敛
- `tracking_vel`: 8 → 13.7，速度追踪恢复到高水平
- `noise_std`: 0.36 → 0.75，探索范围持续扩大

网络在这个阶段找到了速度追踪与步态切换之间的平衡点。

#### 阶段 4: 第二次突变 — 课程扩展冲击 (Step 1000~1200)

| Step | tracking_vel | gait_match | max_cmd_x | noise_std |
|------|-------------|------------|-----------|-----------|
| 1000 | **13.69** | **7.47** | **2.0** | 0.75 |
| 1200 | **9.89** | **4.33** | **2.5** | 0.82 |

**`tracking_vel` 从 13.7 暴跌到 9.9，`gait_match` 从 7.5 暴跌到 4.3**

**原因**: `max_command_x` 从 2.0 跳到 2.5！这是**速度课程自动扩展**触发了（`update_command_curriculum`: 当 `tracking_vel` 均值 > 0.8 × 权重时扩展范围）。突然出现 2.0~2.5 m/s 的高速指令是网络从未见过的，而步态映射规则中 Bound 区间只到 1.5 m/s，2.0~2.5 属于训练时未充分覆盖的区域，导致所有指标大幅回退。

#### 阶段 5: 恢复期 + 相位追踪突破 (Step 1200~2200)

| Step | tracking_vel | command_phase | gait_match | noise_std |
|------|-------------|--------------|------------|-----------|
| 1400 | 11.58 | 6.11 | 6.17 | 0.82 |
| 2000 | 11.28 | 6.54 | 5.84 | **0.90** |
| 2200 | 12.30 | **10.06** | 4.17 | 0.85 |

**`command_phase` 从 6.5 突涨到 10.1** (step 2000→2200)，同时 `gait_match` 从 5.8 下降到 4.2。

**原因**: `noise_std` 在 step 2000 到达峰值 0.90 后开始回落，网络从"探索模式"进入"收敛模式"。相位追踪开始精确化，但代价是步态匹配下降——**网络选择了牺牲步态准确性来换取更好的相位追踪**。这符合奖励权重的优先级：`tracking_command_phase`(15) > `gait_velocity_matching`(10)，网络的选择是理性的。

#### 阶段 6: 最终收敛 (Step 2200~3000)

| Step | tracking_vel | command_phase | gait_match | feasibility | noise_std |
|------|-------------|--------------|------------|-------------|-----------|
| 2400 | 12.46 | 11.86 | 3.37 | 7.52 | 0.79 |
| 2600 | 12.56 | 12.75 | 2.55 | 7.42 | 0.79 |
| 2999 | 12.48 | **13.08** | **2.40** | **7.46** | 0.77 |

趋势已经明确：`command_phase` 持续上升(13.08)，`gait_match` 持续下降(2.40)。网络最终做出了策略权衡：**优先追踪相位和速度，放弃精确步态匹配**。`noise_std` 从峰值 0.90 逐渐回落到 0.77，收敛趋于稳定。

### 4.3 已发现的设计缺陷: 跳跃可行性退化

#### 问题描述

`rew_jump_feasibility_correct = 7.46/8.0 (93.3%)` 看似表现优秀，但实际上是一个**退化解 (degenerate solution)**。

分析 `vel_ratio`（`rew_tracking_vel / 权重`，即原始追踪质量）：

| Step | vel_ratio | feas_ratio | 说明 |
|------|-----------|------------|------|
| 800 | 0.673 | 0.890 | |
| 1000 | **0.685** (峰值) | 0.928 | vel_ratio 全程未超过 0.685 |
| 2000 | 0.567 | 0.870 | 课程扩展后进一步下降 |
| 2999 | 0.624 | 0.933 | |

`vel_ratio` 全程从未超过 0.685，远低于跳跃可行性要求的 **0.85**。

#### 退化机制

可行性检查的 5 个条件是 AND 关系，其中 `vel_tracking_good = vel_reward > 0.85` 几乎永远不满足（因为 High-Level 引入步态切换后平均追踪质量只有 0.62~0.68）。这导致：

- `true_feasible` 在约 93% 的时刻 = 0（不可跳）
- 网络学到的最优策略：**永远输出 feasibility ≈ 0**
- 计算：93% × exp(0) + 7% × exp(-5) ≈ 0.93 = 7.46/8.0，完全吻合

#### 部署后果

按空格跳跃时，`jump_feasibility_score` 始终 < 0.7，**跳跃请求永远被拒绝**。

#### 修复方向

1. **降低阈值重新训练**: `vel_tracking_good` 从 0.85 降到 0.5~0.6，让约 40-50% 时刻 `true_feasible = 1`
2. **部署时绕过检查**: 在 `rl_policy.py` 中将 `jump_feasibility_threshold` 降到 0 或直接跳过可行性检查
3. **重新设计奖励**: 用非对称奖励代替对称 MSE，惩罚"漏报"（条件满足但输出低分）的力度大于"误报"

### 4.4 部署可行性评估

| 功能 | 就绪度 | 说明 |
|------|-------|------|
| 基础运动 | 可用 | 速度追踪(62.4%)和相位追踪(87.2%)表现良好 |
| 步态自动切换 | 部分可用 | 有一定切换能力(24%),但在阶段3曾达74.7%后因课程扩展和权重竞争回退 |
| 跳跃安全检查 | **不可用** | 网络退化为永远输出"不可跳",部署时跳跃请求永远被拒绝 |
| 自适应跳跃高度 | **不可用** | 依赖跳跃信号触发,跳跃本身不可用则高度输出无意义 |

---

## 5. 部署架构

### 5.1 ROS2 节点拓扑

```
键盘控制器 (keyboard_command.py)
  ├── 运动指令 → rl_policy.py 内部
  └── /joy topic → low_level_ctrl.cpp (状态机)

MuJoCo 仿真器 (mujoco_simulator.py)
  ├── /mujoco/lowstate → rl_policy.py (IMU + 关节状态)
  ├── /mujoco/vel → rl_policy.py (基座速度, High-Level需要)
  └── /mujoco/force → rl_policy.py (接触力, High-Level需要)

层级策略节点 (rl_policy.py)
  ├── 加载 High-Level + Low-Level 两个策略模型
  ├── /rl/target_pos → low_level_ctrl.cpp (12维关节目标)
  ├── /rl/gait_id → (调试: 当前步态)
  ├── /rl/jump_signal → (调试: 跳跃信号)
  ├── /rl/jump_feasibility → (调试: 可行性评分)
  └── /rl/adaptive_height → (调试: 自适应高度)

底层电机控制 (low_level_ctrl.cpp)
  ├── /joy → 状态机 (趴下/站起/策略执行)
  └── PD控制器 → /lowcmd → MuJoCo/真机
```

### 5.2 部署配置 (go2.yaml)

```yaml
# Low-Level Policy
policy_path: "../../../../resources/policies/policy_1_Jan28.pt"
num_actions: 16
num_obs: 68
action_scale: 0.25

# High-Level Policy
high_policy_path: "../../../../resources/policies/policy_high_level_16Feb.pt"
num_high_obs: 35
num_high_actions: 3

# 传感器缩放
default_angles: [0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1.0,-1.5,-0.1,1,-1.5]
lin_vel_scale: 2.0
ang_vel_scale: 0.25
dof_pos_scale: 1.0
dof_vel_scale: 0.05
cmd_scale: [2.0, 2.0, 0.25]

# 控制频率
high_decimation: 200    # 5Hz
low_decimation: 10      # 100Hz

# 决策融合
jump_feasibility_threshold: 0.7
enable_debug_output: true
```

### 5.3 启动流程

**键盘模式 (3 个终端)**:
```bash
# 环境设置
source /opt/ros/humble/setup.zsh
source ~/Repo/cpg_jump-main/deploy/src/unitree_ros2/install/setup.zsh
source ~/Repo/cpg_jump-main/deploy/install/setup.zsh

# Terminal 1: MuJoCo 仿真器
ros2 run deploy_rl_policy mujoco_simulator.py

# Terminal 2: 底层电机控制
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true

# Terminal 3: 层级策略节点 + 键盘控制 (保持前台焦点)
ros2 run deploy_rl_policy rl_policy.py --is_simulation True --control keyboard
```

**Xbox 模式 (4 个终端)**:
```bash
# Terminal 1-2: 同上
# Terminal 3: ros2 run joy joy_node
# Terminal 4: ros2 run deploy_rl_policy rl_policy.py --is_simulation True --control xbox
```

### 5.4 键盘操作

| 按键 | 功能 | 说明 |
|------|------|------|
| 1 | 站起 | 对应 Xbox B 键 |
| 2 | 趴下 | 对应 Xbox A 键 |
| 3 | 执行策略 | 对应 Xbox LB+RB |
| ↑↓←→ | 移动 | 长按逐渐加速, 松开自然减速 |
| Q | Walk 演示 | 直接设速 0.3 m/s |
| W | Trot 演示 | 直接设速 1.0 m/s |
| E | Bound 演示 | 直接设速 1.8 m/s |
| Space | 跳跃 | 触发跳跃请求 (需网络可行性>0.7) |
| S | 急停 | 速度归零 |
| R | 转向 | 右旋 |
| ESC | 退出 | — |

**操作流程**: 按 `1` 站起 → 按 `3` 执行策略 → 方向键或 Q/W/E 控制运动 → 空格跳跃

---

## 6. 关键文件清单

### 训练侧

| 文件 | 功能 |
|------|------|
| `legged_gym/envs/base/jump_config.py` | 训练超参数、奖励权重、步态速度阈值 |
| `legged_gym/envs/base/jump_env.py` | 双层环境: High-Level观测/适配层/奖励函数 |
| `legged_gym/envs/base/legged_robot.py` | Low-Level基类: CPG控制/跳跃状态机 |
| `legged_gym/utils/cpg_rl.py` | CPG振荡器实现 |
| `rsl_rl/rsl_rl/modules/actor_critic.py` | Actor-Critic网络 (可学习noise std) |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | 训练循环 (支持strict=False加载) |

### 部署侧

| 文件 | 功能 |
|------|------|
| `deploy/src/deploy_rl_policy/configs/go2.yaml` | 部署配置 (双层策略路径/频率/融合参数) |
| `deploy/src/deploy_rl_policy/scripts/config.py` | YAML加载 (支持High-Level参数) |
| `deploy/src/deploy_rl_policy/scripts/rl_policy.py` | 双层推理节点 (High+Low+CPG+IK) |
| `deploy/src/deploy_rl_policy/scripts/keyboard_command.py` | 键盘控制器 (替代joy_node) |
| `deploy/src/deploy_rl_policy/scripts/cpg_rl.py` | 部署端CPG实现 |
| `deploy/src/deploy_rl_policy/scripts/mujoco_simulator.py` | MuJoCo仿真桥接 |

### 策略模型

| 文件 | 说明 |
|------|------|
| `deploy/resources/policies/policy_1_Jan28.pt` | Low-Level策略 (预训练, 10000轮) |
| `deploy/resources/policies/policy_high_level_16Feb.pt` | High-Level策略 (3000轮) |

---

## 7. 真机部署注意事项

当前系统在 MuJoCo 仿真中验证。真机部署需要额外实现：

1. **速度估计器**: High-Level观测需要基座线速度，真机没有直接传感器。需通过腿部运动学反推或 EKF 估计（项目已有 EKF 实现可参考）
2. **接触力估计**: High-Level观测需要4脚接触状态，真机需通过关节力矩间接估计
3. **通信延迟**: 真机 ROS2 通信可能有延迟，需确保 High-Level 5Hz / Low-Level 100Hz 的频率对齐
4. **安全限制**: 建议先在较低速度下测试，逐步放开速度范围
