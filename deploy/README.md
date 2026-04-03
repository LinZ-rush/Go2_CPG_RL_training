# Deploy-an-RL-policy-on-the-Unitree-Go2-robot
This repository provides a framework for low-level control of a legged robot (Unitree Go2), using ROS 2 as the communication middleware. The MuJoCo simulator is used to validate the control policy in simulation. Once the policy performs well in MuJoCo, you can deploy it on the real robot by simply setting the ROS parameter is_simulation to false. Also a base velocity estimator using extended Karman Filter is provided to estimate the velocity of base. If you find this project useful, please consider giving it a ⭐️ to support development!


## Environment
- **Ubuntu**: 20.04/22.04
- **ROS 2**: Foxy/Humble
- **MuJoCo**: 3.2.3
- **Python**: 3.8.20/3.10
- **Pinocchio**: 3.4.0

## Access Robot Sensor Data via ROS 2
Refer to the official Unitree ROS 2 repository for setup and examples:
[unitree ros2](https://github.com/unitreerobotics/unitree_ros2)

*source unitree_ros2 before building this workspace*

### Potential Build Issue
When building the ROS 2 packages, you may encounter the following error:
```bash
ModuleNotFoundError: No module named 'unitree_go.unitree_go_s__rosidl_typesupport_c’
```
or 
``` bash
rosidl_generator_py.import_type_support_impl.UnsupportedTypeSupport: Could not import 'rosidl_typesupport_c' for package 'unitree_go’
```
This issue occurs because **ROS 2 Foxy supports only Python 3.8** for generating C-based type support modules. If a different Python version is used, the build process may still succeed, but the runtime will **fail to locate the generated files**, resulting in import errors when executing ROS 2 commands.

### Solution
Manually set your Python version to **3.8** when building the workspace. For example:
```bash
export PYTHON_EXECUTABLE=/usr/bin/python3.8
colcon build --symlink-install
```
Make sure Python 3.8 is installed and available at the specified path.




## Simulation
After successfully building this workspace, you can launch the simulation in mujoco with the following commands:
```bash
source install/setup.bash
ros2 run deploy_rl_policy mujoco_simulator.py
``` 
Here is a screenshot of the simulation scene:
<p align="center">
  <img src="./resources/images/mujoco.png" alt="MuJoCo Simulation Scene" width="500"/>
</p>

## Control Logic
The robot's behavior is controlled by a 3-state state machine, including:
* **Laying Down**
* **Standing Up**
* **Executing RL Policy**

### State Transition (XBox Controller)
* **Initial State:** Robot automatically enters "Laying Down" state
* **B Button:** Transitions from "Laying Down" → "Standing Up"
* **A Button:** Transitions from "Standing Up" → "Laying Down"
* **LB + RB Simultaneously:** While standing, executes RL Policy (remains in standing state)

### Notes:
* The "Executing RL Policy" state is considered a special case of the "Standing Up" state
* Controller inputs are only processed when the robot is in the appropriate state for that transition

### Launch Control Nodes

Run the following commands in separate terminals to activate the control system:

```bash
# Terminal 1: XBox Controller Interface
ros2 run joy joy_node

# Terminal 2: State Machine Controller
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true # true: simulation  false: real robot

# Terminal 3: Reinforcement Learning Policy
ros2 run deploy_rl_policy RL_policy.py --is_simulation True  # or False
```
Node Description:
1. joy_node
    * Interfaces with XBox controller hardware
    * Publishes controller input to /joy topic
2. low_level_control
    * Implements the 3-state machine (Laying Down/Standing Up/RL Policy)
    * Handles state transitions based on controller input
    * Sends lowcmd to simulator or the real robot
3. ​​RL_policy.py​​:
    * Executes reinforcement learning policy
    * Activated only in "Executing RL Policy" state (LB+RB pressed while standing)

## Implementing Custom Policies

To use your own reinforcement learning policy with the system:

1. **Modify Policy Path**  
   Edit the policy file path in `RL_policy.py` to point to your custom policy.

2. **Data Sequence Considerations**  
   - The Unitree robot uses a specific joint order that may differ from your training environment
   - Verify your policy's output sequence matches the robot's expected input order

3. **Safety Recommendations**  
   ```diff
   + Always test new policies in simulation first
   - Avoid deploying untested policies directly to hardware
   ```
You can refer to the [official documentation](https://support.unitree.com/home/en/developer/Basic_services) to check the correct joint order.

## Base Velocity Estimator
If your policy requires the base velocity as part of the observation and it's not available from onboard sensors, you can use the **Base Velocity Estimator** to estimate it.

It's implemented as an **Extended Kalman Filter (EKF)** with a measurement model and a system model. The measurement is computed from kinematic equations using [Pinocchio](https://github.com/stack-of-tasks/pinocchio). Here's a rough overview of the theory behind it:
[https://glowing-torch.github.io/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/](https://glowing-torch.github.io/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/).


#本项目的source路径
conda activate go2_ros_env     
source /opt/ros/humble/setup.zsh
source ~/Repo/cpg_jump-main/deploy/src/unitree_ros2/install/setup.zsh
source ~/Repo/cpg_jump-main/deploy/install/setup.zsh



键盘操作提示：

↑↓←→ 移动，长按加速
Q/W/E 一键切到 Walk(0.3) / Trot(1.0) / Bound(1.8) 速度
空格 跳跃
S 急停
ESC 退出

原始 config.py 只读基础字段，而当前 config.py 强制读 high_policy_path。rl_policy_single.py import 的是同一个 config.py，所以当前 config.py 会读 High-Level 字段，但这没关系——go2.yaml 里有这些字段，多读不影响。rl_policy_single.py 代码内不会使用这些字段。

这样就完全兼容了。重新编译后两种模式都可用：

双层层级模式 (当前, 3 个终端):


ros2 run deploy_rl_policy mujoco_simulator.py
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true
ros2 run deploy_rl_policy rl_policy.py --is_simulation True --control keyboard
原始单层模式 (4 个终端):


ros2 run deploy_rl_policy mujoco_simulator.py
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true
ros2 run joy joy_node
ros2 run deploy_rl_policy rl_policy_single.py --is_simulation True

集成分析
两个项目对比:

方面	目标项目	cpg_jump 参考
rl_policy.py	45/270维标准MLP	68维 CPG+RL
cpg_rl.py	无	有
mujoco_simulator.py	有 /mujoco/force 但被注释, 无 /mujoco/vel	两者都有
low_level_ctrl.cpp	与 cpg_jump 完全相同	同左
base_velocity_estimator	有 (真机EKF)	无
集成计划 (6步)
Step 1 — 复制 cpg_rl.py 到目标项目 (无修改，安全操作)

Step 2 — 创建 rl_policy_cpg.py (基于 rl_policy_single.py，改 config 路径)

Step 3 — 创建 configs/go2_cpg.yaml (CPG专用配置)

Step 4 — 修改 mujoco_simulator.py：

取消注释 force 发布，添加坐标变换 (像 cpg_jump 那样用 site_xmat)
新增 /mujoco/vel 发布
时步 0.005 → 0.001 (对齐 CPG 1000Hz)
切换 xml → go2.xml (有速度传感器)
Step 5 — 更新 CMakeLists.txt，加入 cpg_rl.py 和 rl_policy_cpg.py

Step 6 — 复制策略文件 policy_1_Jan28.pt

不修改的文件: rl_policy.py, low_level_ctrl.cpp, base_velocity_estimator/, config.py



读calf和thigh
source /opt/ros/humble/setup.zsh
source ~/Repo/cpg_jump-main/deploy/src/unitree_ros2/install/setup.zsh
python3 ~/Repo/cpg_jump-main/deploy/src/deploy_rl_policy/scripts/thigh_calf_monitor.py
