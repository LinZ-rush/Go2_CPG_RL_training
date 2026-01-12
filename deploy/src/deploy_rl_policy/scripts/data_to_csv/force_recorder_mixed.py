#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int16
import csv
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Training configuration for quadruped gaits")
RECORD_DURATION = 10.0

parser.add_argument(
    "--gait",
    type=str,
    choices=["trot", "bound", "walk", "pronk", "mixed"],
    default="mixed",
    help="Select which gait to train or evaluate: trot, bound, walk, pronk, or mixed."
)
args = parser.parse_args()
gait = args.gait

# 规范化路径
CSV_CONTACT_FILE = f"/home/song/cpg_jump/deploy/csv/gait_contact_forces_10s_{gait}.csv"
CSV_POS_FILE = f"/home/song/cpg_jump/deploy/csv/gait_positions_10s_{gait}.csv"
if gait == "mixed":
    RECORD_DURATION = 30.0
    CSV_CONTACT_FILE = f"/home/song/cpg_jump/deploy/csv/gait_contact_forces_30s_mixed_gaits.csv"
    CSV_POS_FILE = f"/home/song/cpg_jump/deploy/csv/gait_positions_30s_mixed_gaits.csv"

os.makedirs(os.path.dirname(CSV_CONTACT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_POS_FILE), exist_ok=True)

# 话题名
FORCE_TOPIC = "/mujoco/force"
POS_TOPIC = "/mujoco/foot_positions"
GAIT_ID_TOPIC = "/rl/gait_id"   # 新增：步态 ID (Int16)

# 约定 pos_msg.data 顺序为: [FL_x, FL_z, FR_x, FR_z, RL_x, RL_z, RR_x, RR_z]

class GaitDataRecorder(Node):
    def __init__(self):
        super().__init__('gait_data_recorder_node')

        # 新增：当前 gait_id（未收到前用 -1 占位）
        self.current_gait_id = -1
        self.received_gait_id = False

        # 订阅力传感
        self.force_subscription = self.create_subscription(
            Float32MultiArray,
            FORCE_TOPIC,
            self.force_callback,
            10
        )

        # 订阅脚端位置
        self.pos_subscription = self.create_subscription(
            Float32MultiArray,
            POS_TOPIC,
            self.pos_callback,
            10
        )

        # 新增：订阅 gait_id
        self.gait_subscription = self.create_subscription(
            Int16,
            GAIT_ID_TOPIC,
            self.gait_callback,
            10
        )

        # 数据缓冲区
        # contact_buffer: [time, FL_Fz, FR_Fz, RL_Fz, RR_Fz, gait_id]
        self.contact_buffer = []
        # pos_buffer: [time, FL_x, FL_z, FR_x, FR_z, RL_x, RL_z, RR_x, RR_z]
        self.pos_buffer = []

        # 计时起点
        self.start_time = self.get_clock().now()

        # 定时保存退出
        self.timer = self.create_timer(RECORD_DURATION, self.save_and_exit)

        self.get_logger().info(f'Gait data recorder started. Recording for {RECORD_DURATION} seconds.')
        self.get_logger().info(f'Listening: forces="{FORCE_TOPIC}", positions="{POS_TOPIC}", gait_id="{GAIT_ID_TOPIC}".')

    def _now_seconds(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def gait_callback(self, msg: Int16):
        """更新当前步态 ID"""
        self.current_gait_id = int(msg.data)
        if not self.received_gait_id:
            self.received_gait_id = True
            self.get_logger().info(f"First gait_id received: {self.current_gait_id}")

    def force_callback(self, msg: Float32MultiArray):
        """记录力数据：每只脚的 Fz + 当前 gait_id"""
        current_time = self._now_seconds()

        # 假设 force 数据顺序为：FL, FR, RL, RR × (Fx,Fy,Fz) 共12
        if len(msg.data) >= 12:
            try:
                fl_force_z = float(msg.data[2])   # FL Fz
                fr_force_z = float(msg.data[5])   # FR Fz
                rl_force_z = float(msg.data[8])   # RL Fz
                rr_force_z = float(msg.data[11])  # RR Fz
            except Exception as e:
                self.get_logger().warn(f'Error parsing force message: {e}')
                return
        else:
            self.get_logger().warn('Received force message with insufficient data. Skipping this frame.')
            return

        if not self.received_gait_id:
            # 只提示一次即可（可选）
            self.get_logger().warn_once = getattr(self, "warn_once", False)
            if not self.getattr_default("warn_once", False):
                self.get_logger().warn('No gait_id received yet. Using -1 as placeholder.')
                self.warn_once = True

        self.contact_buffer.append([
            current_time,
            fl_force_z, fr_force_z, rl_force_z, rr_force_z,
            self.current_gait_id  # 新增：把 gait_id 写入
        ])

    # 小工具：取属性默认值
    def getattr_default(self, name, default):
        return getattr(self, name) if hasattr(self, name) else default

    def pos_callback(self, msg: Float32MultiArray):
        """记录脚端位置 x/z"""
        current_time = self._now_seconds()

        if len(msg.data) >= 8:
            try:
                fl_x = float(msg.data[0]); fl_z = float(msg.data[1])
                fr_x = float(msg.data[2]); fr_z = float(msg.data[3])
                rl_x = float(msg.data[4]); rl_z = float(msg.data[5])
                rr_x = float(msg.data[6]); rr_z = float(msg.data[7])
            except Exception as e:
                self.get_logger().warn(f'Error parsing position message: {e}')
                return
        else:
            self.get_logger().warn('Received position message with insufficient data. Skipping this frame.')
            return

        self.pos_buffer.append([
            current_time,
            fl_x, fl_z, fr_x, fr_z, rl_x, rl_z, rr_x, rr_z
        ])

    def save_contact_csv(self):
        """保存接触力数据到 CSV（含 gait_id）"""
        if not self.contact_buffer:
            self.get_logger().info('No contact force data received. Skipping contact CSV save.')
            return

        with open(CSV_CONTACT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time (s)',
                'FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz',
                'GaitID'  # 新增列
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
