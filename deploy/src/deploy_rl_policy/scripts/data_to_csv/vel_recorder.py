#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import csv
import os

# ---------- 配置路径 ----------
gait="BOUND"
CSV_VEL_FILE = f"/home/song/cpg_jump/deploy/csv/velocities_10s_{gait}.csv"
CSV_TARGET_VEL_FILE = f"/home/song/cpg_jump/deploy/csv/target_velocities_10s_{gait}.csv"

# 确保目录存在
os.makedirs(os.path.dirname(CSV_VEL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_TARGET_VEL_FILE), exist_ok=True)

# ---------- 录制时长（秒） ----------
RECORD_DURATION = 10.0

# ---------- 话题名 ----------
VEL_TOPIC = '/mujoco/vel'          # 实际速度
TARGET_VEL_TOPIC = '/rl/target_vel' # 目标速度

class VelocityRecorder(Node):
    def __init__(self):
        super().__init__('velocity_recorder_node')

        # 缓冲区
        self.vel_buffer = []         # [time, vel_data...]
        self.target_vel_buffer = []  # [time, target_vel_data...]

        # 计时起点
        self.start_time = self.get_clock().now()

        # 订阅两个速度话题
        self.vel_subscription = self.create_subscription(
            Float32MultiArray,
            VEL_TOPIC,
            self.vel_callback,
            10)

        self.target_vel_subscription = self.create_subscription(
            Float32MultiArray,
            TARGET_VEL_TOPIC,
            self.target_vel_callback,
            10)

        # 定时器：到时保存并退出
        self.timer = self.create_timer(RECORD_DURATION, self.save_and_exit)

        self.get_logger().info(f'Velocity recorder started. Recording for {RECORD_DURATION} seconds.')
        self.get_logger().info(f'Listening for "{VEL_TOPIC}" and "{TARGET_VEL_TOPIC}".')

    def _now_seconds(self):
        """获取当前时间（相对起点的秒数）"""
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def vel_callback(self, msg: Float32MultiArray):
        """处理实际速度话题"""
        current_time = self._now_seconds()
        try:
            data_list = [float(x) for x in msg.data]
        except Exception as e:
            self.get_logger().warn(f'Error parsing /mujoco/vel: {e}')
            return
        self.vel_buffer.append([current_time] + data_list)

    def target_vel_callback(self, msg: Float32MultiArray):
        """处理目标速度话题"""
        current_time = self._now_seconds()
        try:
            data_list = [float(x) for x in msg.data]
        except Exception as e:
            self.get_logger().warn(f'Error parsing /rl/target_vel: {e}')
            return
        self.target_vel_buffer.append([current_time] + data_list)

    def save_vel_csv(self):
        """保存实际速度"""
        if not self.vel_buffer:
            self.get_logger().info('No velocity data received. Skipping save.')
            return

        max_len = max(len(row) for row in self.vel_buffer)
        header = ['Time (s)'] + [f'vel_{i}' for i in range(max_len - 1)]

        with open(CSV_VEL_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in self.vel_buffer:
                row = row + [''] * (max_len - len(row))
                writer.writerow(row)

        self.get_logger().info(f'Velocity data saved to {CSV_VEL_FILE} (rows={len(self.vel_buffer)}).')

    def save_target_vel_csv(self):
        """保存目标速度"""
        if not self.target_vel_buffer:
            self.get_logger().info('No target velocity data received. Skipping save.')
            return

        max_len = max(len(row) for row in self.target_vel_buffer)
        header = ['Time (s)'] + [f'target_vel_{i}' for i in range(max_len - 1)]

        with open(CSV_TARGET_VEL_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in self.target_vel_buffer:
                row = row + [''] * (max_len - len(row))
                writer.writerow(row)

        self.get_logger().info(f'Target velocity data saved to {CSV_TARGET_VEL_FILE} (rows={len(self.target_vel_buffer)}).')

    def save_and_exit(self):
        """定时器触发：保存所有数据并退出"""
        self.get_logger().info(f'Recording duration of {RECORD_DURATION} seconds reached. Saving data and exiting.')
        self.save_vel_csv()
        self.save_target_vel_csv()
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = VelocityRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recording stopped manually. Saving data...")
        node.save_vel_csv()
        node.save_target_vel_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
