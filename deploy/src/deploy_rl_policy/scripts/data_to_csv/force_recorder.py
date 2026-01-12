#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import csv
import os
import time
import os

import argparse

parser = argparse.ArgumentParser(description="Training configuration for quadruped gaits")
RECORD_DURATION = 10.0

parser.add_argument(
    "--gait",
    type=str,
    choices=["trot", "bound", "walk", "pronk","mixed"],
    default="trot",
    help="Select which gait to train or evaluate: trot, bound, walk, or pronk."
)
args = parser.parse_args()
gait = args.gait
# 规范化路径
CSV_CONTACT_FILE = f"/home/song/cpg_jump/deploy/csv/gait_contact_forces_10s_{gait}.csv"
CSV_POS_FILE = f"/home/song/cpg_jump/deploy/csv/gait_positions_10s_{gait}.csv"
if gait=="mixed":
    RECORD_DURATION = 30.0
    CSV_CONTACT_FILE = f"/home/song/cpg_jump/deploy/csv/gait_contact_forces_30s_mixed_gaits.csv"
    CSV_POS_FILE = f"/home/song/cpg_jump/deploy/csv/gait_positions_30s_mixed_gaits.csv"
    
os.makedirs(os.path.dirname(CSV_CONTACT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_POS_FILE), exist_ok=True)



# 录制时长 (秒)

# 位置话题与位置数据顺序约定（若实际不同，请修改）
POS_TOPIC = '/mujoco/foot_positions'
# 约定 pos_msg.data 顺序为: [FL_x, FL_z, FR_x, FR_z, RL_x, RL_z, RR_x, RR_z]

class GaitDataRecorder(Node):
    def __init__(self):
        super().__init__('gait_data_recorder_node')

        # 订阅力传感（用于接触判定），话题保持你原来的
        self.force_subscription = self.create_subscription(
            Float32MultiArray,
            '/mujoco/force',
            self.force_callback,
            10)

        # 订阅脚端位置（x, z）
        self.pos_subscription = self.create_subscription(
            Float32MultiArray,
            POS_TOPIC,
            self.pos_callback,
            10)

        # 数据缓冲区
        # 现在 contact_buffer 存储原始接触力（浮点数），并不做阈值判断
        self.contact_buffer = []   # 存储 [time, fl_Fz, fr_Fz, rl_Fz, rr_Fz]
        self.pos_buffer = []       # 存储 [time, FL_x, FL_z, FR_x, FR_z, RL_x, RL_z, RR_x, RR_z]

        # 计时起点
        self.start_time = self.get_clock().now()

        # 创建定时器，在指定时长后保存并退出
        self.timer = self.create_timer(RECORD_DURATION, self.save_and_exit)

        self.get_logger().info(f'Gait data recorder started. Recording for {RECORD_DURATION} seconds.')
        self.get_logger().info(f'Listening for force on "/mujoco/force" and positions on "{POS_TOPIC}".')

    def _now_seconds(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def force_callback(self, msg: Float32MultiArray):
        """处理力数据回调：只记录每只脚的垂直分量（Fz），不在此处判断接触"""
        current_time = self._now_seconds()

        # 假设 force 数据顺序为：前左(FL), 前右(FR), 后左(RL), 后右(RR) 且每只脚对应 XYZ（共12）
        if len(msg.data) >= 12:
            fl_force_z = float(msg.data[2])   # FL 脚 Z 轴力
            fr_force_z = float(msg.data[5])   # FR 脚 Z 轴力
            rl_force_z = float(msg.data[8])   # RL 脚 Z 轴力
            rr_force_z = float(msg.data[11])  # RR 脚 Z 轴力
        else:
            self.get_logger().warn('Received force message with insufficient data. Skipping this frame.')
            return

        # 不做阈值判断，直接存储原始力值
        self.contact_buffer.append([
            current_time,
            fl_force_z,
            fr_force_z,
            rl_force_z,
            rr_force_z
        ])

    def pos_callback(self, msg: Float32MultiArray):
        """处理脚端位置回调，保存 x/z 到 pos_buffer"""
        current_time = self._now_seconds()

        if len(msg.data) >= 8:
            # 约定顺序: [FL_x, FL_z, FR_x, FR_z, RL_x, RL_z, RR_x, RR_z]
            try:
                fl_x = float(msg.data[0])
                fl_z = float(msg.data[1])
                fr_x = float(msg.data[2])
                fr_z = float(msg.data[3])
                rl_x = float(msg.data[4])
                rl_z = float(msg.data[5])
                rr_x = float(msg.data[6])
                rr_z = float(msg.data[7])
            except Exception as e:
                self.get_logger().warn(f'Error parsing position message: {e}')
                return
        else:
            self.get_logger().warn('Received position message with insufficient data. Skipping this frame.')
            return

        self.pos_buffer.append([
            current_time,
            fl_x, fl_z,
            fr_x, fr_z,
            rl_x, rl_z,
            rr_x, rr_z
        ])

    def save_contact_csv(self):
        """保存接触力数据到 CSV（原始力值）"""
        if not self.contact_buffer:
            self.get_logger().info('No contact force data received. Skipping contact CSV save.')
            return

        with open(CSV_CONTACT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time (s)',
                'FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz'
            ])
            writer.writerows(self.contact_buffer)

        self.get_logger().info(f'Contact force data saved to {CSV_CONTACT_FILE} (rows={len(self.contact_buffer)}).')

    def save_pos_csv(self):
        """保存位置数据到 CSV"""
        if not self.pos_buffer:
            self.get_logger().info('No position data received. Skipping positions CSV save.')
            return

        with open(CSV_POS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time (s)',
                'FL_x', 'FL_z',
                'FR_x', 'FR_z',
                'RL_x', 'RL_z',
                'RR_x', 'RR_z'
            ])
            writer.writerows(self.pos_buffer)

        self.get_logger().info(f'Position data saved to {CSV_POS_FILE} (rows={len(self.pos_buffer)}).')

    def save_and_exit(self):
        """定时器回调：保存所有数据并优雅退出"""
        self.get_logger().info(f'Recording duration of {RECORD_DURATION} seconds reached. Saving data and exiting.')
        self.save_contact_csv()
        self.save_pos_csv()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GaitDataRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().info("Recording stopped manually. Saving data...")
        node.save_contact_csv()
        node.save_pos_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
