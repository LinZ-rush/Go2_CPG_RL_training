#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int16
import csv
import os

# ---------- 配置 ----------
GAIT_NAME = {0: "Trot", 1: "Walk", 2: "Bound", 3: "Pronk"}  # 可选：不用也行
gait_tag = "mixed"  # 仅用于文件名区分
CSV_COMBINED_FILE = f"/home/song/cpg_jump/deploy/csv/vel_target_gait_30s_{gait_tag}.csv"

os.makedirs(os.path.dirname(CSV_COMBINED_FILE), exist_ok=True)

# ---------- 录制时长（秒） ----------
RECORD_DURATION = 30.0

# ---------- 话题名 ----------
VEL_TOPIC = '/mujoco/vel'            # 实际速度（Float32MultiArray）
TARGET_VEL_TOPIC = '/rl/target_vel'  # 目标速度（Float32MultiArray）
GAIT_ID_TOPIC = '/rl/gait_id'        # 步态ID（Int16）

class VelocityRecorder(Node):
    def __init__(self):
        super().__init__('velocity_recorder_node')

        # 缓存最新值
        self.latest_vel = []           # list[float]
        self.latest_target_vel = []    # list[float]
        self.latest_gait_id = -1       # int，占位
        self.has_gait_id = False

        # 记录“已观察到的最大长度”（用于保存表头与补齐空列）
        self.max_len_vel = 0
        self.max_len_tvel = 0

        # 行缓冲：每次任一话题到达就推一行
        # 行格式： [time, vel..., target_vel..., gait_id]
        self.rows = []

        # 时间起点
        self.start_time = self.get_clock().now()

        # 订阅三个话题
        self.vel_sub = self.create_subscription(Float32MultiArray, VEL_TOPIC, self.vel_callback, 10)
        self.tvel_sub = self.create_subscription(Float32MultiArray, TARGET_VEL_TOPIC, self.target_vel_callback, 10)
        self.gait_sub = self.create_subscription(Int16, GAIT_ID_TOPIC, self.gait_id_callback, 10)

        # 定时器：到时保存并退出（create_timer 为周期调用；第一次触发即保存并退出）
        self.timer = self.create_timer(RECORD_DURATION, self.save_and_exit)

        self.get_logger().info(f"Velocity recorder started. Recording for {RECORD_DURATION} seconds.")
        self.get_logger().info(f'Listening: "{VEL_TOPIC}", "{TARGET_VEL_TOPIC}", "{GAIT_ID_TOPIC}".')

    def _now_seconds(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    # ====== 回调：更新最新值，并立刻记录一行 ======
    def vel_callback(self, msg: Float32MultiArray):
        try:
            self.latest_vel = [float(x) for x in msg.data]
            self.max_len_vel = max(self.max_len_vel, len(self.latest_vel))
        except Exception as e:
            self.get_logger().warn(f'Error parsing {VEL_TOPIC}: {e}')
            return
        self._append_row(trigger="vel")

    def target_vel_callback(self, msg: Float32MultiArray):
        try:
            self.latest_target_vel = [float(x) for x in msg.data]
            self.max_len_tvel = max(self.max_len_tvel, len(self.latest_target_vel))
        except Exception as e:
            self.get_logger().warn(f'Error parsing {TARGET_VEL_TOPIC}: {e}')
            return
        self._append_row(trigger="target_vel")

    def gait_id_callback(self, msg: Int16):
        self.latest_gait_id = int(msg.data)
        self.has_gait_id = True
        self._append_row(trigger="gait_id")

    # ====== 记录一行 ======
    def _append_row(self, trigger: str):
        t = self._now_seconds()
        # 复制当前快照，避免后续被原地修改影响
        vel = list(self.latest_vel) if self.latest_vel else []
        tvel = list(self.latest_target_vel) if self.latest_target_vel else []
        gid = self.latest_gait_id if self.has_gait_id else -1

        self.rows.append([t, vel, tvel, gid])
        # 可选调试输出：
        # self.get_logger().debug(f"Row appended by {trigger}: t={t:.3f}, len(vel)={len(vel)}, len(tvel)={len(tvel)}, gid={gid}")

    # ====== 保存 ======
    def save_combined_csv(self):
        if not self.rows:
            self.get_logger().info("No data rows recorded. Skip saving.")
            return

        # 统一表头：Time + vel_0.. + target_vel_0.. + GaitID
        header = ['Time (s)']
        header += [f'vel_{i}' for i in range(self.max_len_vel)]
        header += [f'target_vel_{i}' for i in range(self.max_len_tvel)]
        header += ['GaitID']

        with open(CSV_COMBINED_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for t, vel, tvel, gid in self.rows:
                # 按最大长度补齐
                vel_padded = vel + [''] * (self.max_len_vel - len(vel))
                tvel_padded = tvel + [''] * (self.max_len_tvel - len(tvel))
                writer.writerow([t] + vel_padded + tvel_padded + [gid])

        self.get_logger().info(f'Combined data saved to {CSV_COMBINED_FILE} (rows={len(self.rows)}).')

    def save_and_exit(self):
        self.get_logger().info(f"Recording duration {RECORD_DURATION}s reached. Saving and exiting.")
        self.save_combined_csv()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = VelocityRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recording stopped manually. Saving data...")
        node.save_combined_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
