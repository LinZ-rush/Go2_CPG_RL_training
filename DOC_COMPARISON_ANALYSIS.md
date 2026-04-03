# Config 参数与奖励函数对比：cpg_jump vs cpg_jump-main

两个项目架构相同，均使用 `legged_robot.py` + `legged_robot_config.py` + `go2_config.py`，
任务均为步态切换 + 跳跃。cpg_jump-main 是在原版基础上将步态限定为 Trot，并人工分两阶段训练。

---

## 一、Config 参数对比

### 1.1 `legged_robot_config.py` — `commands` 类

| 参数 | cpg_jump | cpg_jump-main | 说明 |
|------|---------|--------------|------|
| `is_jump` | `1` | `0`（Phase 1），手动改为 `1`（Phase 2）| 控制状态机是否允许触发跳跃 |
| `freq_max` | `40` rad/s | `31.4` rad/s | CPG 角频率上限 |
| `freq_low` | `3.14`（正） | `-3.14`（**允许负频率**） | 负频率允许 CPG 反向旋转 |
| `loc_height` | `0.32` m | `0.25` m | 行走时目标机身高度 |
| `jump_up_height` | `[0.4, 0.6]` m | `[0.45, 0.6]` m | 跳跃高度指令范围 |
| `cpg_couple` | 无 | `True`（strength=1.0）| main 版新增腿间相位耦合 |

### 1.2 `legged_robot_config.py` — `rewards.scales` 类

| 奖励项 | cpg_jump | cpg_jump-main | 说明 |
|-------|---------|--------------|------|
| `tracking_vel` | `10` | `10` | 相同 |
| `jump_sig` | `10` | `10` | 相同（函数逻辑不同，见第二节）|
| `vel_diff` | `-0.05` | `-0.05` | 相同 |
| `orientation` | `-100` | `-100` | 相同 |
| `collision` | `-1` | `-1` | 相同 |
| `feet_contact_forces` | `-0.01` | `-0.01` | 相同 |
| `energy` | `-0.0015` | `-0.0015` | 相同 |
| `action_rate` | `0`（关） | `-0.01`（**开**）| main 版惩罚动作突变 |
| `dof_pos_limits` | `0`（关） | `-10.0`（**开**）| main 版惩罚关节超限 |
| `feet_slip` | **无此项** | `-5`（**新增**）| main 版惩罚支撑相脚掌滑动 |
| `base_height` | `0`（关） | `-5.0`（**开**）| main 版惩罚机身高度偏差 |
| `termination` | `0`（关） | `-1.0`（**开**）| main 版惩罚非超时终止 |
| `cpg_freq` | `0`（关） | `-0.01`（**开**）| main 版惩罚 Trot 频率超阈值 |
| `feet_air_time` | `0`（关） | `1.0`（**开**）| main 版奖励腿部腾空时间 |
| `jump_up` | `0`（预留）| 无此项 | 原版预留，未使用 |
| `gait_matching` | `0`（预留）| 无此项 | 原版预留，未使用 |

### 1.3 `legged_robot_config.py` — 其他奖励参数

| 参数 | cpg_jump | cpg_jump-main | 说明 |
|------|---------|--------------|------|
| `base_height_target` | `0.3` m | `0.25` m | 与 `loc_height` 一致 |
| `soft_dof_pos_limit` | `1.0`（无软限）| `0.9` | main 版留 10% 安全余量 |
| `height_sigma` | 无 | `0.1` | main 版行走高度奖励独立 sigma |

### 1.4 `go2_config.py` — PD 控制器

| 参数 | cpg_jump | cpg_jump-main | 说明 |
|------|---------|--------------|------|
| `stiffness` | `100` N·m/rad | `50` N·m/rad | main 版减半，更柔顺 |
| `damping` | `2` N·m·s/rad | `1` N·m·s/rad | main 版减半，减少震荡 |

### 1.5 `legged_robot_config.py` — PPO runner

| 参数 | cpg_jump | cpg_jump-main | 说明 |
|------|---------|--------------|------|
| `max_iterations` | `3000` | `5000` | main 版更长 |
| `save_interval` | `50` | `500` | main 版保存更稀疏 |

---

## 二、奖励函数定义对比（`legged_robot.py`）

### 2.1 相同的函数（逻辑完全一致）

"是否启用"取决于 config 中对应 scale 是否非零（scale=0 时 `_prepare_reward_function` 跳过该项）。

| 函数 | 原理 | cpg_jump | cpg_jump-main |
|-----|-----|---------|--------------|
| `_reward_tracking_vel` | `exp(-‖v_cmd - v_base‖² / σ)`，指数衰减速度跟踪误差 | ✅ `10` | ✅ `10` |
| `_reward_vel_diff` | `‖v_cmd - v_base‖²`，速度误差平方（与 tracking_vel 互补，线性惩罚） | ✅ `-0.05` | ✅ `-0.05` |
| `_reward_orientation` | `‖g_proj_xy‖²`，重力投影水平分量，惩罚机身倾斜 | ✅ `-100` | ✅ `-100` |
| `_reward_energy` | `Σ|τ · dq̇|`，各关节功率之和，惩罚能量消耗 | ✅ `-0.0015` | ✅ `-0.0015` |
| `_reward_collision` | 惩罚接触体列表（thigh/calf）上的碰撞次数 | ✅ `-1` | ✅ `-1` |
| `_reward_feet_contact_forces` | `Σ(‖F_foot‖ - F_max).clip(0)`，惩罚超出阈值的脚底接触力 | ✅ `-0.01` | ✅ `-0.01` |
| `_reward_feet_air_time` | 奖励腾空时长 > 0.5s 的首次落地，零速度指令时不给奖励 | ❌ `0` | ✅ `1.0` |
| `_reward_base_height` | `(h - h_target)²`，惩罚机身高度偏离目标 | ❌ `0` | ✅ `-5.0` |
| `_reward_action_rate` | `Σ(a_t - a_{t-1})²`，惩罚相邻时刻动作突变 | ❌ `0` | ✅ `-0.01` |
| `_reward_termination` | 非超时终止（摔倒）时给惩罚，超时不惩罚 | ❌ `0` | ✅ `-1.0` |
| `_reward_dof_pos_limits` | `Σ(q - q_lim).clip`，惩罚关节位置超软限 | ❌ `0` | ✅ `-10.0` |
| `_reward_dof_acc` | `Σ((dq̇/dt))²`，惩罚关节加速度突变 | ❌ `0` | ❌ `0` |
| `_reward_lin_vel_z` | `v_z²`，惩罚机身垂直方向速度（跳跃时不适合开启）| ❌ `0` | ❌ `0` |
| `_reward_ang_vel_xy` | `Σω_xy²`，惩罚机身横滚/俯仰角速度 | ❌（scales 中无此项）| ❌（scales 中无此项）|
| `_reward_stand_still` | 零速度指令时惩罚关节偏离默认角度，防止原地乱动 | ❌ `0` | ❌ `0` |
| `_reward_stumble` | 脚端水平接触力 > 5× 垂直接触力时惩罚，防止撞墙 | ❌ `0` | ❌ `0` |

### 2.2 仅 cpg_jump-main 新增的函数

**`_reward_feet_slip`**
```python
def _reward_feet_slip(self):
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    feet_vel = self.rb_states[:, self.feet_indices, 7:10]
    return torch.sum(feet_vel**2 * contact.unsqueeze(-1), dim=[1, 2])
```
接地时脚端速度越大惩罚越重，强制支撑相脚掌静止。原版无此函数。

### 2.3 同名但实现不同的函数

#### `_reward_cpg_freq` / `_reward_cpg_frequency`

**cpg_jump**（名为 `_reward_cpg_frequency`，系数=0 未启用）：
```python
def _reward_cpg_frequency(self):
    cpg_freq = torch.mean(self._cpg.X_dot[:, 0, :], dim=1)
    return cpg_freq  # 直接返回频率均值，无惩罚逻辑
```

**cpg_jump-main**（名为 `_reward_cpg_freq`，系数=-0.01）：
```python
def _reward_cpg_freq(self):
    freq = self._cpg.X_dot[:, 1, :]           # 取相位频率（X_dot[:,1,:]）
    low_freq_gait = ~(self.commands[:, 4].long() == 2)  # 非 Bound 步态
    freq_penalty = torch.zeros_like(self.commands[:, 4])
    freq_penalty[low_freq_gait] = torch.sum(
        (freq[low_freq_gait, :] - 18.84).clip(min=0.) ** 2, dim=1
    )
    return freq_penalty  # 超过 18.84 rad/s 才惩罚
```
差异：main 版读取 `X_dot[:,1,:]`（相位维），设步态相关阈值，超阈值才惩罚；原版读 `X_dot[:,0,:]`（振幅维），无实际意义。

---

#### `_reward_jump_sig`（最核心的差异）

两版函数结构相同，均包含：行走奖励 → Loco→Jump 触发 → Jump→Loco 切换，但细节不同：

**行走阶段奖励**（完全相同）：
```python
jump_rwd[loco] += 0.95 * exp(-phase_error / jump_sigma)   # CPG 相位匹配
jump_rwd[loco] += 0.05 * exp(-height_error / sigma)       # 机身高度
```

**跳跃阶段关节速度惩罚系数**：

| | cpg_jump | cpg_jump-main |
|--|---------|--------------|
| 系数 | `1e-4` | `2e-4`（翻倍）|

**Loco→Jump 触发条件差异**：

```python
# cpg_jump：无 is_jump 守卫，始终可能触发
loco_goal = (~jump_sig) & (vel_reward > 0.85) & feet_in_contact \
          & (phase_error.mean() < 0.05) & (cooldown == 0) \
          & (mode_timer > min_mode_steps) \
          & (episode_buf % (0.5*max_len) > 0.25*max_len)

# cpg_jump-main：加了 is_jump==1 守卫，Phase 1 时恒为 False
loco_goal = (self.cfg.commands.is_jump == 1) & (~jump_sig) & (vel_reward > 0.85) \
          & feet_in_contact & (phase_error.mean() < 0.05) & (cooldown == 0) \
          & (mode_timer > min_mode_steps) \
          & (episode_buf % (0.5*max_len) > 0.25*max_len)
```

**Jump→Loco 切换（着陆检测）差异**：

| 行为 | cpg_jump | cpg_jump-main |
|-----|---------|--------------|
| 大奖触发时机 | **空中**达到目标高度（`reached_height & ~feet_in_contact`）| **着陆后**（`landing_edge & max_jump_height > 0.42`）|
| 额外着陆约束 | 无 | `orientation_error < 0.1`（姿态合格才给大奖）|
| 着陆等待机制 | `waiting_for_landing` 标志 + `landing_counter > 3` | 无，检测到 `landing_edge` 即切换 |
| 着陆后速度 | 固定重置为 `0.3 m/s` | 乘以 `0.6`（比例衰减）|
