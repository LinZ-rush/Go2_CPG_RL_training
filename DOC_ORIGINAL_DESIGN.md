# 原始项目设计文档

## 1. 项目概述

本项目基于 Unitree Go2 四足机器人，实现了基于 **CPG（中枢模式发生器）+ 强化学习**的运动跳跃控制系统。项目采用 Isaac Gym 进行大规模并行训练，MuJoCo 进行 Sim2Sim 验证，最终部署到真机。

整体流程: `Train (Isaac Gym)` → `Play (验证)` → `Sim2Sim (MuJoCo)` → `Sim2Real (真机)`

---

## 2. 系统架构

### 2.1 训练侧 (legged_gym + rsl_rl)

原始系统采用**单层控制架构**：

```
命令输入 (速度/步态/跳跃) → Low-Level Policy (68→16维) → CPG+IK → 关节力矩
```

**核心组件**：

| 组件 | 文件 | 功能 |
|------|------|------|
| 训练入口 | `legged_gym/scripts/train.py` | 启动 PPO 训练 |
| 验证入口 | `legged_gym/scripts/play.py` | 可视化策略并导出 TorchScript |
| 环境基类 | `legged_gym/envs/base/legged_robot.py` | 单层运动环境，包含步态状态机、跳跃逻辑 |
| CPG 控制器 | `legged_gym/utils/cpg_rl.py` | 中枢模式发生器，生成节律运动 |
| 神经网络 | `rsl_rl/rsl_rl/modules/actor_critic.py` | Actor-Critic 网络 (MLP) |
| PPO 算法 | `rsl_rl/rsl_rl/algorithms/ppo.py` | 近端策略优化 |
| 训练循环 | `rsl_rl/rsl_rl/runners/on_policy_runner.py` | 数据收集 + 网络更新 |

### 2.2 部署侧 (deploy)

```
XBox 手柄 → joy_node → /joy topic
                           ↓
               low_level_ctrl.cpp (状态机: 趴下/站起/策略执行)
                           ↓
MuJoCo ← /lowcmd ← PD控制器 ← /rl/target_pos ← rl_policy.py (单层策略)
  ↓
/mujoco/lowstate, /mujoco/vel, /mujoco/force → rl_policy.py
```

**ROS2 节点**：

| 节点 | 功能 |
|------|------|
| `joy_node` | ROS2 标准手柄驱动，发布 `/joy` topic |
| `low_level_ctrl` | C++ 状态机 + PD 电机控制器 |
| `rl_policy.py` | 加载单个策略模型，运行推理 |
| `mujoco_simulator.py` | MuJoCo 仿真桥接，发布传感器数据 |

---

## 3. 控制原理

### 3.1 CPG (中枢模式发生器)

CPG 是一种生物启发的运动控制方法，通过耦合振荡器产生节律性运动模式。每条腿对应一个振荡器，通过相位差实现不同步态：

| 步态 | ID | 相位模板 [FL, FR, RL, RR] | 特点 |
|------|----|--------------------------|------|
| Trot | 0 | [0.0, 0.5, 0.5, 0.0] | 对角同步，中速稳定 |
| Walk | 1 | [0.0, 0.5, 0.75, 0.25] | 四拍步态，低速平稳 |
| Bound | 2 | [0.0, 0.0, 0.5, 0.5] | 前后同步，高速奔跑 |
| Pronk | 3 | [0.0, 0.0, 0.0, 0.0] | 四足同步，跳跃专用 |

CPG 状态包含 4 组 (r, θ) 振荡器，RL 策略输出 16 维偏移量叠加到 CPG 基础运动上，经逆运动学 (IK) 转换为关节目标位置。

### 3.2 单层控制流程 (legged_robot.py)

原始 `legged_robot.py` 中的 `_reward_jump_sig()` 函数集成了完整的运动-跳跃状态机：

1. **运动模式**: 根据 `commands[:, 4]` 的步态 ID 追踪相位目标
2. **跳跃触发**: 满足 5 个条件时自动触发跳跃
   - 速度追踪 > 85%
   - 相位误差 < 5%
   - 四足接触地面
   - 冷却时间已过
   - 当前模式保持 > min_mode_steps
3. **跳跃执行**: 追踪目标跳跃高度 (0.4~0.8m)
4. **着陆恢复**: 检测全脚触地后切回运动模式

### 3.3 Low-Level Policy (68→16维)

**输入观测 (68维)**:
```
[0:3]   base_ang_vel * scale        # 角速度
[3:6]   projected_gravity           # 重力投影
[6:9]   commands * cmd_scale        # 速度指令
[9]     jump_height                 # 跳跃高度
[10]    gait_id                     # 步态ID
[11]    jump_signal                 # 跳跃信号
[12:24] (dof_pos - default) * scale # 关节位置
[24:36] dof_vel * scale             # 关节速度
[36:52] last_actions                # 上一步动作
[52:68] CPG state (r, θ, ṙ, θ̇)    # CPG振荡器状态
```


---

## 4. 训练配置

### 4.1 原始奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `tracking_vel` | 10.0 | 速度追踪 (核心) |
| `tracking_command_phase` | 10.0 | 相位追踪 (核心) |
| `orientation_roll` | -200 | 防侧翻 |
| `orientation_yaw` | -300 | 防偏航 |
| `collision` | -1 | 碰撞惩罚 |
| `action_rate` | -0.01 | 动作平滑 |
| `energy` | -0.001 | 能耗惩罚 |

### 4.2 训练参数

- 环境数: 2048
- 仿真时步: 0.001s (1000Hz)
- 策略时步: Low-Level 每 10 步 (100Hz)
- Episode 长度: 20s
- PPO 学习率: 1e-3 (自适应调度)
- 网络结构: MLP [512, 256, 128]

---

## 5. 部署流程 (原始)

### 5.1 启动步骤 (4 个终端)

```bash
# 环境设置
source /opt/ros/humble/setup.zsh
source ~/Repo/cpg_jump-main/deploy/src/unitree_ros2/install/setup.zsh
source ~/Repo/cpg_jump-main/deploy/install/setup.zsh

# Terminal 1: MuJoCo 仿真器
ros2 run deploy_rl_policy mujoco_simulator.py

# Terminal 2: 底层电机控制
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true

# Terminal 3: 手柄驱动
ros2 run joy joy_node

# Terminal 4: RL 策略
ros2 run deploy_rl_policy rl_policy.py --is_simulation True
```

### 5.2 Xbox 手柄控制

| 按键 | 功能 |
|------|------|
| B | 趴下 → 站起 |
| A | 站起 → 趴下 |
| LB + RB | 执行 RL 策略 |
| 十字键上下 | 前进/后退 (长按加速) |
| 十字键左右 | 左右平移 |
| LT + RT | 退出 |

### 5.3 部署配置 (go2.yaml 原始)

```yaml
policy_path: "../../../../resources/policies/policy_1.pt"
num_actions: 16
num_obs: 68
action_scale: 0.25
default_angles: [0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1.0,-1.5,-0.1,1,-1.5]
ang_vel_scale: 0.25
dof_pos_scale: 1.0
dof_vel_scale: 0.05
cmd_scale: [2.0, 2.0, 0.25]
```

---

## 6. 项目依赖

| 组件 | 版本 |
|------|------|
| Ubuntu | 20.04 / 22.04 |
| ROS 2 | Foxy / Humble |
| MuJoCo | 3.2.3 |
| Python | 3.8 / 3.10 |
| Isaac Gym | Preview 4 |
| PyTorch | 1.x / 2.x |

---

## 7. 关键局限性

原始单层架构存在以下问题，这也是后续改进的动机：

1. **步态固定**: 步态 ID 由外部命令直接指定，不能根据速度自动切换
2. **跳跃触发硬编码**: 跳跃条件在 `legged_robot.py` 中硬编码（速度追踪>85%、相位<5%等），无法学习更灵活的判断
3. **跳跃高度固定**: 跳跃目标高度从命令中直接获取，不能根据运动状态自适应调整
4. **人机耦合不足**: 跳跃完全由系统自主触发（满足条件即跳），操作员无法控制跳跃时机
