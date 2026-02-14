🚀 项目描述：Unitree Go2 分层 CPG-RL 跳跃控制 (Hierarchical CPG-RL Jumping)
1. 项目概况 (Overview)
本项目是一个基于 Isaac Gym 的分层强化学习框架，用于控制 Unitree Go2 四足机器人实现多步态切换与动态跳跃。
核心架构采用 Teacher-Student / High-Low 双层策略：

底层 (Low-Level / Spinal Cord): 基于 CPG (Central Pattern Generator) 的稳健运动控制器。

顶层 (High-Level / Brain): 负责决策跳跃时机、高度和步态类型的宏观控制器。

2. 核心架构与数据流 (Architecture & Data Flow)
A. High-Level Policy (正在训练)

任务: 决策层 (task=jump)

文件: jump_env.py, jump_config.py

输入 (Obs): 35维 (宏观状态：Base Vel, Gravity, Commands, CPG state等)。

输出 (Action): 3维 (归一化到 [0,1])，分别代表：

Jump Height Ratio (跳跃高度比例)

Gait ID Ratio (步态选择比例)

Jump Signal (跳跃触发信号)

B. Adapter Layer (关键逻辑 - 不要修改!)

位置: jump_env.py -> compute_low_level_observations()

功能: 将 High-Level 的 3维 Action 语义映射并拼接为 Low-Level 需要的 68维 Observation。

映射规则:

Jump Height = Action[0] * 0.3 + 0.3 (物理范围 0.3m-0.6m)

Gait ID = floor(Action[1] * 3.99) (映射为整数 0,1,2,3)

Jump Sig = Action[2] > 0.5 (布尔开关)

维度对齐: 必须严格凑齐 68维，且 Low Actions 必须补齐为 16维。

C. Low-Level Policy (已冻结/预训练)

任务: 执行层 (task=go2)

文件: legged_robot.py, go2_config.py

输入 (Obs): 68维 (严禁包含 Privileged Info 如 Base Linear Velocity 或 Contact)。

输出 (Action): 16维 (CPG 参数：Amplitude, Frequency, Offsets 等)。

执行: CPG 参数 -> 正弦发生器 -> 逆运动学 (IK) -> PD 控制 -> 电机力矩。

3. 关键状态与约束 (Constraints)
Low-Level 模型: 使用导出的 policy_1.pt (JIT 格式)，其输入层固定为 68 节点。

CPG 参数: 动作输出对应 16 维参数，其中包含 4 维 X-direction offset。

Sim2Real: 最终目标是部署到 Go2 真机，通过 Xbox 手柄控制 High-Level 的输入 (Commands)。

4. 已修复的问题 (History Context)
维度不匹配: 修复了 JumpEnv 曾错误地生成 74 维观测向量（包含多余线速度和接触力）导致无法输入给 68 维 Low-Level 网络的问题。

语义映射: 增加了显式的语义映射层，防止 High-Level 的归一化输出直接干扰 Low-Level 的物理定义。