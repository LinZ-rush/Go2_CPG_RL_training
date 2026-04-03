# 修改记录

本文档记录从原始单层控制到当前双层层级控制的所有具体代码和功能变更。

---

## 1. 架构变更概览

| 方面 | 原始 | 当前 |
|------|------|------|
| 控制架构 | 单层 (Low-Level only) | 双层 (High-Level + Low-Level) |
| 步态选择 | 外部命令指定 | 网络自动根据速度选择 |
| 跳跃触发 | 硬编码条件自动触发 | 操作员请求 + 网络可行性检查 |
| 跳跃高度 | 从指令直接获取 | 网络自适应调整 [0.3, 0.6]m |
| 输入设备 | Xbox 手柄 (4节点) | 键盘/Xbox 可切换 (3/4节点) |
| High-Level 输出语义 | 无 | [0]步态 [1]可行性 [2]高度 |

---

## 2. 训练侧修改

### 2.1 jump_config.py

**文件**: `legged_gym/envs/base/jump_config.py`

**修改 1: 控制频率参数化**
```python
# 新增
class control:
    low_decimation = 10     # Low-Level每10步 (100Hz)
    high_decimation = 200   # High-Level每200步 (5Hz)
```

**修改 2: 新增3个奖励权重**
```python
class rewards.scales:
    tracking_vel = 20           # 原10 → 20 (提高优先级)
    tracking_command_phase = 15 # 原10 → 15 (提高优先级)
    gait_velocity_matching = 10.0   # 新增: 步态-速度匹配
    jump_feasibility_correct = 8.0  # 新增: 跳跃可行性判断
    smooth_gait_transition = -2.0   # 新增: 惩罚频繁切换
```

**修改 3: 新增步态-速度映射阈值**
```python
gait_speed_thresholds = {
    'walk_to_trot': 0.6, 'trot_to_walk': 0.4,
    'trot_to_bound': 1.6, 'bound_to_trot': 1.4,
}
```

**修改 4: 新增跳跃可行性阈值**
```python
jump_feasibility_thresholds = {
    'vel_tracking_min': 0.85, 'phase_error_max': 0.05,
    'orientation_error_max': 0.1, 'min_mode_steps': 100,
    'cooldown_steps': 150,
}
```

**修改 5: PPO 超参数调整**
```python
init_noise_std = 0.3       # 原0.1 → 0.3 (3维输出需更大探索)
entropy_coef = 0.005       # 降低,防止noise_std失控
num_steps_per_env = 24     # 原12 → 24 (更好的梯度估计)
desired_kl = 0.01          # 恢复保守更新
```

**修改 6: 加载路径指定**
```python
load_run = 'Feb15_22-45-06_'  # 原-1 → 指定运行名 (修复字母排序bug)
checkpoint = 3000              # 原-1 → 指定3000轮
```

---

### 2.2 jump_env.py

**文件**: `legged_gym/envs/base/jump_env.py`

**修改 1: step() 函数 — 双层控制循环**
- 原始: 直接执行 low_actions
- 当前: `high_actions → sigmoid → compute_low_level_observations → low_actions`
- High-Level 每 200 步执行一次，内部循环 20 次 Low-Level (每次 10 步)

```python
def step(self, actions):
    high_actions_scaled = torch.sigmoid(actions)  # 新增: sigmoid映射
    self.actions = high_actions_scaled
    for _ in range(high_decimation / low_decimation):  # 20次
        self.compute_low_level_observations(high_actions_scaled)
        self.low_actions = policy(self.low_level_obs_buf)
        for _ in range(low_decimation):  # 10次
            self.torques = self._compute_torques(self.low_actions)
            # ... 物理仿真 ...
```

**修改 2: compute_low_level_observations() — 适配层重写**

原始 (注释掉): High-Level 直接拼入观测
当前: 3 维 sigmoid 输出 → 物理语义映射

```python
# [0]: 步态ID (floor(x * 3.99) → {0,1,2,3})
gait_cmd = torch.floor(high_actions[:, 0:1] * 3.99)

# [1]: 跳跃信号 (请求 AND 可行性>0.7)
jump_request = (torch.rand(...) < 0.1).float()
jump_signal_cmd = (jump_request * (jump_feasibility > 0.7)).float()

# [2]: 跳跃高度 (x * 0.3 + 0.3 → [0.3, 0.6]m)
jump_height_cmd = high_actions[:, 2:3] * 0.3 + 0.3
```

**修改 3: 新增 _reward_gait_velocity_matching() (~55行)**
- 根据当前速度判断最优步态 (Walk/Trot/Bound)
- 过渡区域 (0.4~0.6, 1.4~1.6) 容忍两种步态
- 严格对齐 GAIT_TEMPLATES ID: 0=TROT, 1=WALK, 2=BOUND

**修改 4: 新增 _reward_jump_feasibility_correct() (~80行)**
- 计算 5 个真实可行性条件
- 网络预测与真值的误差 → 奖励 = exp(-error/0.2)

**修改 5: 新增 _reward_smooth_gait_transition() (~25行)**
- 检测步态ID变化，变化时返回 1.0 (由负权重变为惩罚)

**修改 6: _init_buffers() — low_actions 维度**
```python
self.low_actions = torch.zeros(self.num_envs, 16, ...)  # 原15 → 16
```

---

### 2.3 actor_critic.py

**文件**: `rsl_rl/rsl_rl/modules/actor_critic.py`

**修改 1: noise std 改为可学习参数**
```python
# 原始: 固定噪声
# self.std = init_noise_std * torch.ones(num_actions, requires_grad=False)

# 当前: 可学习参数
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
```

**修改 2: std 范围限制**
```python
def update_distribution(self, observations):
    mean = self.actor(observations)
    std_clamped = torch.clamp(self.std, min=0.01, max=1.0)  # 防止失控
    self.distribution = Normal(mean, mean*0. + std_clamped)
```

---

### 2.4 on_policy_runner.py

**文件**: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

**修改: strict=False 加载**
```python
# 原始: 严格加载
self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

# 当前: 宽松加载 (兼容旧checkpoint缺少std key)
self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
```

---

## 3. 部署侧修改

### 3.1 go2.yaml

**文件**: `deploy/src/deploy_rl_policy/configs/go2.yaml`

新增配置项:
```yaml
# High-Level Policy
high_policy_path: "../../../../resources/policies/policy_high_level_16Feb.pt"
num_high_obs: 35
num_high_actions: 3

# 传感器缩放 (新增)
lin_vel_scale: 2.0

# 控制频率 (新增)
high_decimation: 200
low_decimation: 10

# 决策融合 (新增)
jump_feasibility_threshold: 0.7
enable_debug_output: true
```

---

### 3.2 config.py

**文件**: `deploy/src/deploy_rl_policy/scripts/config.py`

新增加载项:
```python
self.high_policy_path = config["high_policy_path"]
self.num_high_obs = config["num_high_obs"]
self.num_high_actions = config["num_high_actions"]
self.lin_vel_scale = config["lin_vel_scale"]
self.high_decimation = config.get("high_decimation", 200)
self.low_decimation = config.get("low_decimation", 10)
self.jump_feasibility_threshold = config.get("jump_feasibility_threshold", 0.7)
self.enable_debug_output = config.get("enable_debug_output", False)
```

---

### 3.3 rl_policy.py (完全重写)

**文件**: `deploy/src/deploy_rl_policy/scripts/rl_policy.py`

**原始**: 单层策略推理节点，加载一个策略模型
**当前**: 双层层级控制节点，完全重写

核心变更:

| 功能 | 原始 | 当前 |
|------|------|------|
| 策略模型 | 1个 (Low-Level) | 2个 (High + Low) |
| 节点名 | `data_receiver` | `hierarchical_policy_node` |
| 输入控制 | 仅 Xbox | Xbox / 键盘 可切换 (`--control` 参数) |
| 推理循环 | 统一频率 | 分频: High 5Hz + Low 100Hz |
| 速度订阅 | 无 | 订阅 `/mujoco/vel` (High-Level需要) |
| 接触状态 | 标量 | 4维向量 (High-Level需要) |
| 跳跃逻辑 | 无 | 决策融合 (按键+可行性+接触) |
| ROS topic | `/rl/target_pos`, `/rl/gait_id` | 新增 `/rl/jump_feasibility`, `/rl/adaptive_height` |
| CPG输入 | torch.clip(actions) | torch.from_numpy(...) 显式转换 |

关键函数:
- `_run_high_level()`: 35维观测构建 → 推理 → sigmoid → 适配层 → 跳跃决策融合
- `_run_low_level()`: 68维观测构建 → 推理 → CPG + IK → 发布target_pos
- `_publish_info()`: 发布步态、跳跃、可行性等调试信息

---

### 3.4 keyboard_command.py (新增文件)

**文件**: `deploy/src/deploy_rl_policy/scripts/keyboard_command.py`

全新文件，替代 Xbox 手柄 + joy_node:

- 后台线程监听终端键盘输入 (tty.setcbreak)
- 20Hz 定时发布 Joy 消息到 `/joy` topic (供 low_level_ctrl.cpp 状态机使用)
- 按键映射: 1→站起(B), 2→趴下(A), 3→执行策略(LB+RB)
- 运动控制: 方向键加速, Q/W/E 步态演示速度, 空格跳跃
- 自然减速: 松开按键 0.2s 后逐渐减速到 0
- 兼容接口: `jump_pressed()`, `is_pressed()`, `is_exit_requested()`, `stop()`

---

### 3.5 CMakeLists.txt

**文件**: `deploy/src/deploy_rl_policy/CMakeLists.txt`

新增安装:
```cmake
install(PROGRAMS
    scripts/keyboard_command.py  # 新增
    ...
)
```

---

## 4. Bug 修复记录

| 问题 | 原因 | 修复 |
|------|------|------|
| play.py 加载 Jan28 而非 Feb15 模型 | `get_load_path` 字母排序 "Jan" > "Feb" | `load_run = 'Feb15_22-45-06_'` 指定路径 |
| `Missing key 'std' in state_dict` | 旧 checkpoint 无可学习 std | `strict=False` 加载 |
| `KeyError: 'policy_path'` | go2.yaml 所有 policy_path 被注释 | 取消注释 policy_1_Jan28.pt |
| `TypeError: clip() got numpy` | actions_scaled 是 numpy 传给 torch.clip | `torch.from_numpy(...).unsqueeze(0)` 转换 |
| `DeprecationWarning: torch→numpy` | CPG tensor 直接赋给 numpy 数组 | 添加 `.numpy()` 显式转换 |
| `No executable found` | ROS2 Python 脚本需 `.py` 后缀 | `ros2 run ... rl_policy.py` |
| 键盘模式无法站起/趴下 | low_level_ctrl.cpp 依赖 /joy topic | keyboard_command.py 发布 Joy 消息 |

---

## 5. 新增策略模型

| 模型 | 训练轮数 | 来源 |
|------|---------|------|
| `policy_high_level_16Feb.pt` | 3000 | Feb15 训练 → play.py 导出 |
| `policy_1_Jan28.pt` | 10000 | Low-Level 预训练 (未变更) |

---

## 6. 未变更的文件

以下文件在本次改动中**未修改**:

- `deploy/src/deploy_rl_policy/scripts/mujoco_simulator.py` — 已有 /mujoco/vel 和 /mujoco/force 发布
- `deploy/src/deploy_rl_policy/src/low_level_ctrl.cpp` — 仅接收 /rl/target_pos 和 /joy
- `deploy/src/deploy_rl_policy/scripts/xbox_command.py` — 保持原有手柄控制
- `legged_gym/envs/base/legged_robot.py` — Low-Level 基类未变更
- `legged_gym/utils/cpg_rl.py` — CPG 核心逻辑未变更
- `rsl_rl/rsl_rl/algorithms/ppo.py` — PPO 算法未变更
