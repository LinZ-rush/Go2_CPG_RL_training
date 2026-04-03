#!/usr/bin/env python3
"""
双层层级控制节点 (Hierarchical CPG-RL Policy)

架构:
  High-Level (大脑, 5Hz):  35维观测 → 3维决策 (步态/跳跃可行性/跳跃高度)
  Adapter (适配层):        3维决策 → 步态ID、跳跃信号、跳跃高度
  Low-Level (脊髓, 100Hz): 68维观测 → 16维CPG参数
  CPG + IK (1000Hz):       CPG参数 → 关节目标位置

控制频率 (与训练对齐):
  - 物理/定时器:  1000Hz (1ms)
  - High-Level:   每200步 = 5Hz   (high_decimation=200)
  - Low-Level:    每10步  = 100Hz (low_decimation=10)

用法:
  键盘控制: python rl_policy.py --control keyboard --is_simulation True
  手柄控制: python rl_policy.py --control xbox --is_simulation True
"""
from cpg_rl import CPG_RL
import rclpy
import torch
from config import Config
import numpy as np
import os
import sys
from rclpy.node import Node
import argparse
from pathlib import Path
from unitree_go.msg import LowState
from std_msgs.msg import Float32MultiArray, Int16, Float32
from collections import deque
import time


PHYSICS_DT = 0.001
EXPECTED_HIGH_DECIMATION = 200
EXPECTED_LOW_DECIMATION = 10


class asset:
    hip_link_length_go2 = 0.0955
    thigh_link_length_go2 = 0.213
    calf_link_length_go2 = 0.213


project_root = Path(__file__).parents[4]


class dataReciever(Node):
    def __init__(self, config: Config):
        super().__init__("hierarchical_policy_node")
        self.config = config
        self.control_mode = args.control  # 'xbox' or 'keyboard'
        self._cpg = CPG_RL(time_step=0.001, num_envs=1, rl_task_string="CPG_OFFSETX")

        # ===== 输入控制器 =====
        if self.control_mode == 'keyboard':
            from keyboard_command import KeyboardController
            self.controller = KeyboardController(self)
        else:
            from xbox_command import XboxController
            self.controller = XboxController(self)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # ===== 加载 High-Level Policy (新增) =====
        high_policy_path = os.path.join(script_dir, config.high_policy_path)
        self.high_policy = torch.jit.load(high_policy_path)
        self.high_policy.eval()
        self.get_logger().info(f"High-Level Policy loaded from {high_policy_path}")

        # ===== 加载 Low-Level Policy (原有) =====
        low_policy_path = os.path.join(script_dir, config.policy_path)
        self.low_policy = torch.jit.load(low_policy_path)
        self.low_policy.eval()
        self.get_logger().info(f"Low-Level Policy loaded from {low_policy_path}")

        # ===== 观测和动作缓冲区 =====
        self.high_obs = np.zeros(config.num_high_obs, dtype=np.float32)          # 35维
        self.low_obs = np.zeros(config.num_obs, dtype=np.float32)                # 68维
        self.high_actions = np.zeros(config.num_high_actions, dtype=np.float32)  # 3维 (sigmoid后)
        self.low_actions = np.zeros(config.num_actions, dtype=np.float32)        # 16维

        # ===== 传感器状态变量 =====
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.cmd = np.array([0.0, 0, 0])
        self.low_state = LowState()
        self.base_lin_vel_cache = np.zeros(3, dtype=np.float32)
        self.contact_cache = np.zeros(4, dtype=np.float32)  # FL, FR, RL, RR

        # ===== 决策融合状态变量 =====
        self.current_gait_id = 0          # 初始步态: Trot (GAIT_TEMPLATES: 0=TROT)
        self.jump_feasibility_score = 0.0
        self.jump_signal = 0
        self.adaptive_jump_height = 0.3
        self.prev_jump_pressed = 0
        self.prev_jump_signal = 0

        # ===== 计数器和状态 =====
        self.count = 0
        self.contact_deque = deque(maxlen=30)
        self.force_contact = True
        self.landing = True
        self.mode_timer = 0
        self.step_start = time.perf_counter()

        # ===== ROS2 订阅 =====
        if args.simulation:
            self.low_state_sub = self.create_subscription(
                LowState, "/mujoco/lowstate", self.low_state_callback, 10)
            self.vel_sub = self.create_subscription(
                Float32MultiArray, "/mujoco/vel", self.vel_callback, 10)
            print("reading data from simulation")
        else:
            self.low_state_sub = self.create_subscription(
                LowState, "/lowstate", self.low_state_callback, 10)
            print("reading data from reality")

        self.force_sub = self.create_subscription(
            Float32MultiArray, "/mujoco/force", self.force_callback, 10)

        # ===== ROS2 发布器 =====
        self.gait_id_puber = self.create_publisher(Int16, "/rl/gait_id", 10)
        self.target_vel_puber = self.create_publisher(Float32MultiArray, "/rl/target_vel", 10)
        self.target_pos_puber = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self.jump_sig_puber = self.create_publisher(Int16, "/rl/jump_signal", 10)
        self.jump_feasibility_puber = self.create_publisher(Float32, "/rl/jump_feasibility", 10)
        self.adaptive_height_puber = self.create_publisher(Float32, "/rl/adaptive_height", 10)

        self.get_logger().info(f"Control mode: {self.control_mode}")
        self.get_logger().info("Waiting for data")
        self.timer = self.create_timer(0.001, self.run)

    # ===== 回调函数 =====

    def low_state_callback(self, msg: LowState):
        self.low_state = msg

    def vel_callback(self, msg: Float32MultiArray):
        """接收MuJoCo发布的基座速度 (High-Level观测需要)"""
        if len(msg.data) >= 3:
            self.base_lin_vel_cache = np.array(msg.data[:3], dtype=np.float32)

    def force_callback(self, msg: Float32MultiArray):
        """接收MuJoCo发布的接触力，计算4脚接触状态"""
        if len(msg.data) >= 12:
            fl_force_z = msg.data[2]
            fr_force_z = msg.data[5]
            rl_force_z = msg.data[8]
            rr_force_z = msg.data[11]
        else:
            self.get_logger().warn('Received force message with insufficient data.')
            return

        fl_contact = 1.0 if fl_force_z < -30 else 0.0
        fr_contact = 1.0 if fr_force_z < -30 else 0.0
        rl_contact = 1.0 if rl_force_z < -30 else 0.0
        rr_contact = 1.0 if rr_force_z < -30 else 0.0

        # 4维接触状态 (供High-Level观测使用)
        self.contact_cache = np.array([fl_contact, fr_contact, rl_contact, rr_contact], dtype=np.float32)

        # 标量接触判断 (供跳跃状态机使用)
        total_contact = fl_contact + fr_contact + rl_contact + rr_contact
        self.force_contact = total_contact >= 1
        self.landing = total_contact >= 4

    # ===== 主控制循环 (1000Hz) =====

    def run(self):
        # --- 退出检查 ---
        if self.control_mode == 'keyboard':
            if self.controller.is_exit_requested():
                self.get_logger().info("Exit requested via keyboard")
                self.controller.stop()
                sys.exit()
        else:
            if self.controller.axes and self.controller.axes[2] == -1 and self.controller.axes[5] == -1:
                sys.exit()

        # --- Step 1: 读取传感器数据 ---
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # comment these two lines for simulation
        self.qj = self.qj[sequence]
        self.dqj = self.dqj[sequence]

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = self.get_gravity_orientation(quat)

        # --- Step 2: 输入处理 ---
        if self.control_mode == 'keyboard':
            # 键盘模式: 速度由 KeyboardController 内部管理
            self.cmd = np.array([self.controller.linear_x,
                                 self.controller.linear_y,
                                 self.controller.angular_z])
        else:
            # Xbox 模式: 原有加速逻辑
            self.left_button, self.right_button = self.controller.is_pressed()
            if self.left_button and self.right_button:
                if self.controller.axes[7] == 0:
                    self.controller.linear_x += -np.sign(self.controller.linear_x) * 0.001
                else:
                    self.controller.linear_x += np.sign(self.controller.axes[7]) * 0.002
                if self.controller.axes[6] == 0:
                    self.controller.linear_y += -np.sign(self.controller.linear_y) * 0.001
                else:
                    self.controller.linear_y += np.sign(self.controller.axes[6]) * 0.002
                self.controller.linear_x = np.clip(self.controller.linear_x, -self.controller.max_speed, self.controller.max_speed)
                self.controller.linear_y = np.clip(self.controller.linear_y, -self.controller.max_speed, self.controller.max_speed)
                self.controller.angular_z = self.controller.get_right_stick()
            else:
                self.controller.linear_x += -np.sign(self.controller.linear_x) * 0.002
                self.controller.linear_y += -np.sign(self.controller.linear_y) * 0.002
                self.controller.angular_z += -np.sign(self.controller.angular_z) * 0.002
            self.cmd = np.array([self.controller.linear_x,
                                 self.controller.linear_y,
                                 self.controller.angular_z])

        # --- Step 3: 计算CPG状态 (两层共用, 转numpy避免torch→numpy警告) ---
        cpg_r = ((self._cpg.X[:, 0, :] - ((self._cpg.mu_up[0] + self._cpg.mu_low[0]) / 2)) * self.config.dof_pos_scale).numpy()
        cpg_theta = ((self._cpg.X[:, 1, :] - np.pi) * 1 / np.pi).numpy()
        cpg_r_dot = (self._cpg.X_dot[:, 0, :] * 1 / 30).numpy()
        cpg_theta_dot = ((self._cpg.X_dot[:, 1, :] - 15) * 1 / 30).numpy()

        # --- Step 4: High-Level Policy (每200步 = 5Hz) ---
        if self.count % self.config.high_decimation == 0:
            self._run_high_level(ang_vel, gravity_orientation, cpg_r, cpg_theta, cpg_r_dot, cpg_theta_dot)

        # --- Step 5: Low-Level Policy + CPG + IK (每10步 = 100Hz) ---
        if self.count % self.config.low_decimation == 0:
            self._run_low_level(ang_vel, gravity_orientation, cpg_r, cpg_theta, cpg_r_dot, cpg_theta_dot)

        # --- Step 6: 发布信息和更新计数 ---
        self._publish_info()
        self.count += 1
        self.mode_timer += 1

    # ===== High-Level推理 (5Hz) =====

    def _run_high_level(self, ang_vel, gravity_orientation, cpg_r, cpg_theta, cpg_r_dot, cpg_theta_dot):
        """
        High-Level Policy推理

        观测格式 (35维, 严格对齐 jump_env.py compute_observations):
          [0:3]   base_lin_vel * lin_vel_scale      (仿真从/mujoco/vel获取)
          [3:6]   base_ang_vel * ang_vel_scale
          [6:9]   projected_gravity
          [9:12]  commands * cmd_scale
          [12:15] last_high_actions (sigmoid后)
          [15:19] contact (4脚, bool→float)
          [19:23] cpg_r
          [23:27] cpg_theta
          [27:31] cpg_r_dot
          [31:35] cpg_theta_dot
        """
        self.high_obs[0:3] = self.base_lin_vel_cache * self.config.lin_vel_scale
        self.high_obs[3:6] = ang_vel * self.config.ang_vel_scale
        self.high_obs[6:9] = gravity_orientation
        self.high_obs[9:12] = self.cmd * self.config.cmd_scale
        self.high_obs[12:15] = self.high_actions
        self.high_obs[15:19] = self.contact_cache
        self.high_obs[19:23] = cpg_r[0]
        self.high_obs[23:27] = cpg_theta[0]
        self.high_obs[27:31] = cpg_r_dot[0]
        self.high_obs[31:35] = cpg_theta_dot[0]

        # 推理
        high_obs_tensor = torch.from_numpy(self.high_obs).unsqueeze(0)
        with torch.no_grad():
            high_output_raw = self.high_policy(high_obs_tensor).detach().numpy()[0]

        # Sigmoid映射到[0,1] (与训练对齐: torch.sigmoid(actions))
        self.high_actions = 1.0 / (1.0 + np.exp(-high_output_raw))

        # ===== 适配层: 3维 → 物理语义 =====

        # [Action 0]: 步态ID (与训练对齐: floor(x * 3.99) → {0,1,2,3})
        # GAIT_TEMPLATES: 0=TROT, 1=WALK, 2=BOUND, 3=PRONK
        self.current_gait_id = int(np.floor(self.high_actions[0] * 3.99))

        # [Action 1]: 跳跃可行性评分 [0, 1]
        self.jump_feasibility_score = float(self.high_actions[1])

        # [Action 2]: 跳跃高度 (与训练对齐: x * 0.3 + 0.3 → [0.3, 0.6]m)
        self.adaptive_jump_height = float(self.high_actions[2] * 0.3 + 0.3)

        # ===== 跳跃决策融合 (按键请求 + 网络检查) =====
        current_pressed = self.controller.jump_pressed()
        current_contact = self.force_contact
        self.contact_deque.append(current_contact)

        # 检测按键上升沿 (从未按到按下)
        jump_triggered = current_pressed and not self.prev_jump_pressed

        if jump_triggered and current_contact:
            if self.jump_feasibility_score > self.config.jump_feasibility_threshold:
                self.jump_signal = 1
                self.mode_timer = 0
                self.get_logger().info(
                    f"JUMP ALLOWED | feasibility={self.jump_feasibility_score:.2f} | "
                    f"height={self.adaptive_jump_height:.2f}m | gait={self.current_gait_id}")
            else:
                self.get_logger().warn(
                    f"JUMP DENIED | feasibility={self.jump_feasibility_score:.2f} < "
                    f"{self.config.jump_feasibility_threshold}")

        # 跳跃结束判定: 落地后超过500步复位
        if current_contact and self.prev_jump_signal and self.mode_timer > 500:
            self.jump_signal = 0

        self.prev_jump_pressed = current_pressed
        self.prev_jump_signal = self.jump_signal

    # ===== Low-Level推理 + CPG + IK (100Hz) =====

    def _run_low_level(self, ang_vel, gravity_orientation, cpg_r, cpg_theta, cpg_r_dot, cpg_theta_dot):
        """
        Low-Level Policy推理

        观测格式 (68维, 严格对齐 jump_env.py compute_low_level_observations):
          [0:3]   base_ang_vel * ang_vel_scale
          [3:6]   projected_gravity
          [6:9]   commands * cmd_scale
          [9:10]  jump_height   (适配层输出)
          [10:11] gait_id       (适配层输出)
          [11:12] jump_signal   (融合结果)
          [12:24] (dof_pos - default) * dof_pos_scale
          [24:36] dof_vel * dof_vel_scale
          [36:52] last_low_actions
          [52:56] cpg_r
          [56:60] cpg_theta
          [60:64] cpg_r_dot
          [64:68] cpg_theta_dot
        """
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale

        self.low_obs[0:3] = ang_vel * self.config.ang_vel_scale
        self.low_obs[3:6] = gravity_orientation
        self.low_obs[6:9] = self.cmd * self.config.cmd_scale
        self.low_obs[9] = self.adaptive_jump_height              # 网络自适应高度
        self.low_obs[10] = self.current_gait_id                  # 网络选择的步态
        self.low_obs[11] = self.jump_signal                      # 融合后的跳跃信号
        self.low_obs[12:24] = qj_obs
        self.low_obs[24:36] = dqj_obs
        self.low_obs[36:52] = self.low_actions
        self.low_obs[52:56] = cpg_r[0]
        self.low_obs[56:60] = cpg_theta[0]
        self.low_obs[60:64] = cpg_r_dot[0]
        self.low_obs[64:68] = cpg_theta_dot[0]
        self.low_obs = np.clip(self.low_obs, -100, 100)

        # Low-Level推理
        low_obs_tensor = torch.from_numpy(self.low_obs).unsqueeze(0)
        with torch.no_grad():
            self.low_actions = self.low_policy(low_obs_tensor).detach().numpy()[0]
            self.low_actions = np.clip(self.low_actions, -100.0, 100.0)

        # CPG + IK → 关节目标位置 (CPG需要tensor输入)
        actions_scaled = torch.from_numpy(self.low_actions * self.config.action_scale).unsqueeze(0)
        des_joint_pos = torch.zeros(1, 12)
        xs, ys, zs = self._cpg.get_CPG_RL_actions(actions_scaled, 31.4, -3.14)
        sideSign = np.array([-1, 1, -1, 1])
        foot_y = torch.ones(1, requires_grad=False) * asset.hip_link_length_go2
        LEG_INDICES = np.array([1, 0, 3, 2])

        for ig_idx, i in enumerate(LEG_INDICES):
            x = xs[:, i]
            z = zs[:, i]
            y = sideSign[i] * foot_y + ys[:, i]
            des_joint_pos[:, 3 * i:3 * i + 3] = self._cpg.compute_inverse_kinematics(asset, i, x, y, z)

        self.target_dof_pos = des_joint_pos.numpy().squeeze()

        # 发布目标关节位置 (重排序为硬件顺序)
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        msg = Float32MultiArray()
        msg.data.extend(self.target_dof_pos.astype(np.float32)[sequence].tolist())
        self.target_pos_puber.publish(msg)

    # ===== 信息发布 =====

    def _publish_info(self):
        """发布步态ID、跳跃信号和调试信息"""
        # 步态ID
        gait_msg = Int16()
        gait_msg.data = self.current_gait_id
        self.gait_id_puber.publish(gait_msg)

        # 跳跃信号
        jump_msg = Int16()
        jump_msg.data = self.jump_signal
        self.jump_sig_puber.publish(jump_msg)

        # 速度指令
        vel_msg = Float32MultiArray()
        vel_msg.data = self.cmd.tolist()
        self.target_vel_puber.publish(vel_msg)

        # 调试信息 (可行性 + 自适应高度)
        if self.config.enable_debug_output:
            feasibility_msg = Float32()
            feasibility_msg.data = self.jump_feasibility_score
            self.jump_feasibility_puber.publish(feasibility_msg)

            height_msg = Float32()
            height_msg.data = self.adaptive_jump_height
            self.adaptive_height_puber.publish(height_msg)

    @staticmethod
    def get_gravity_orientation(quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation


def main():
    rclpy.init()
    reciever_node = dataReciever(config=config)
    try:
        rclpy.spin(reciever_node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(reciever_node, 'controller') and hasattr(reciever_node.controller, 'stop'):
            reciever_node.controller.stop()
        reciever_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # Load config
    config_path = project_root / "src" / "deploy_rl_policy" / "configs" / "go2.yaml"
    config = Config(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_simulation', type=str, choices=["True", "False"], default="True")
    parser.add_argument('--control', type=str, choices=["xbox", "keyboard"], default="keyboard",
                        help="Control mode: xbox (gamepad) or keyboard")
    args = parser.parse_args()
    args.simulation = args.is_simulation == "True"
    main()
