# 训练计划与现状分析

> 更新日期：2026-03-15

---

## 1. 当前训练配置总结

### 目标
单步态（TROT）+ 跳跃（PRONK模板），取代原四步态切换+跳跃方案，减少步态切换抖动。

### 关键参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `gait_id` | 固定 0（TROT） | `commands[:,4] = 0.` |
| `lin_vel_x` | [0, 2.0] m/s | 速度课程范围 |
| `freq_low` | 12.6 rad/s | omega物理下界，防止塌缩，中心=22.0 |
| `freq_max` | 31.4 rad/s | omega物理上界 |
| `is_jump` | 1 | 跳跃始终开启 |
| `add_noise` | False | 观测噪声关闭 |
| `init_noise_std` | 1.0（固定） | 动作探索噪声，非可学习参数 |

---

## 2. 噪声与域随机化分析

### 域随机化（已开启）
```
randomize_friction = True   → 摩擦系数 [0.5, 1.25]
randomize_base_mass = True  → 附加质量 [-1, 6] kg
push_robots = False         → 外力扰动未开启
```

### 噪声设计现状

| 类型 | 状态 | 作用 |
|------|------|------|
| 观测噪声 | **关闭** | — |
| 动作噪声（探索） | **开启，std=1.0固定** | PPO探索机制，不随训练变化 |
| 域随机化 | **开启（摩擦/质量）** | 提供环境级别鲁棒性 |

### 这种设计是否合理？

**当前阶段（sim→sim验证）：合理。**

- 域随机化（摩擦+质量）已提供基础鲁棒性
- 关闭观测噪声可减少干扰，让policy更快收敛到正确步态和跳跃逻辑
- 固定std=1.0在早期保持充分探索，`schedule='adaptive'`会通过调整学习率间接控制更新幅度

**为sim2real准备时需要补充：**
- 建议在policy收敛后，开启 `add_noise = True`（使用默认噪声幅度）做微调训练
- 建议开启 `push_robots = True` 增强外力鲁棒性
- 可以将 `std` 改为 `nn.Parameter`（可学习），让policy自主收敛到更低的动作噪声

### 是否需要调整学习率？

**不需要手动调整。** 当前配置：
```
learning_rate = 1e-3
schedule = 'adaptive'   → 根据KL散度自动调整lr
desired_kl = 0.01       → KL目标，超过则缩小lr
```
`adaptive`调度会在动作噪声大时（KL大）自动降低学习率，无需手动干预。

---

## 3. 跳跃触发机制说明

`loco_goal`条件（[legged_robot.py:899](legged_gym/envs/base/legged_robot.py#L899)）：

```python
loco_goal = (~jump_sig)
          & (vel_reward > 0.85)           ← per-env，逐环境判断
          & feet_in_contact
          & (loco_phase_error[ids].mean() < 0.05)  ← 全局均值，整体训练进度门控
          & (cooldown_timer == 0)
          & (mode_timer > min_mode_steps)
```

**注意**：`loco_phase_error.mean()` 是全局标量门控——一旦整体训练phase收敛（均值<0.05），
任何此时vel_reward>0.85的个体环境即触发跳跃。wandb显示的tracking_vel均值低并不妨碍
部分环境个体触发跳跃。

---

## 4. 已知的sim2sim部署Bug（待修复）

文件：[deploy/src/deploy_rl_policy/scripts/rl_policy.py](deploy/src/deploy_rl_policy/scripts/rl_policy.py)

### Bug 1：rl_task_string不匹配（第55行）
```python
# 现状（错误）：
self._cpg = CPG_RL(..., rl_task_string="CPG_OFFSETX")

# 应为：
self._cpg = CPG_RL(..., rl_task_string="P")
```
训练用`'P'`（actions[8:16]未使用），部署用`"CPG_OFFSETX"`导致未训练的输出被解释为足迹轨迹参数 → 步态失常。

### Bug 2：仿真模式下关节重排序错误（第188-191行）
```python
sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# comment these two lines for simulation  ← 注释说要跳过，但代码没有判断
self.qj = self.qj[sequence]    # 仿真模式下错误地打乱了关节顺序
self.dqj = self.dqj[sequence]
```
修复：在这两行外加 `if not args.simulation:` 保护。

---

## 5. 训练推进路径

### 阶段一：当前训练（进行中）
- **目标**：TROT步态收敛 + 跳跃触发正常
- **验证指标（wandb）**：
  - `Episode/cpg_omega` 从22开始，稳定在16~17
  - `Episode/loco_phase_error` 下降并低于0.05
  - `Episode/jump_fraction` 出现并上升
  - `Loss/surrogate` 正常下降
- **成功标准**：jump_fraction稳定 > 0.1，loco_phase_error < 0.05

### 阶段二：修复sim2sim部署Bug
1. `rl_policy.py` 第55行：`rl_task_string="CPG_OFFSETX"` → `"P"`
2. `rl_policy.py` 第188-191行：添加 `if not args.simulation:` 保护
3. 导出当前最佳policy，在MuJoCo中验证：
   - TROT步态是否正常（无小跳）
   - 跳跃键是否响应

### 阶段三（可选，sim2real准备）
- 开启 `add_noise = True`，在已收敛policy上微调
- 开启 `push_robots = True`，训练外力鲁棒性
- 将 `std` 改为 `nn.Parameter` 可学习，观察policy是否自主降低动作噪声

---

## 6. 速度课程路径（原作者方案）

原作者通过调整 `freq_low / freq_max / lin_vel_x` 分段训练，始终四步态+跳跃同时训练：

| 阶段 | freq_low | freq_max | lin_vel_x | 说明 |
|------|----------|----------|-----------|------|
| 低速 | -3.14 | 18.84 | [0, 1.0] | 基础步态建立 |
| 高速 | -3.14 | 31.4 | [0, 2.0] | 在低速policy上继续训练 |

本次方案改动：固定TROT + `freq_low=12.6`防止omega塌缩，一次性训练[0,2.0]速度范围。
