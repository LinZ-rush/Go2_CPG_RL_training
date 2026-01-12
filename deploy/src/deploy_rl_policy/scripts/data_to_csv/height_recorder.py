#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int16
import csv
import os
import argparse

# ---------------- Args ----------------
parser = argparse.ArgumentParser(description="Record height and jump signal into one CSV.")
parser.add_argument(
    "--gait",
    type=str,
    choices=["trot", "walk", "bound", "pronk", "mixed"],
    default="trot",
    help="Gait tag for this recording (only used for file naming and CSV column)."
)

args = parser.parse_args()

GAIT_TAG = args.gait.upper()
RECORD_DURATION = 5

# ---------------- Paths ----------------
CSV_DIR = "/home/song/cpg_jump/deploy/csv"
os.makedirs(CSV_DIR, exist_ok=True)
CSV_COMBINED_FILE = os.path.join(CSV_DIR, f"height_jump_{int(RECORD_DURATION)}s_{GAIT_TAG}.csv")

# ---------------- Topics ----------------
HEIGHT_TOPIC = "/mujoco/height"     # std_msgs/Float32
JUMP_TOPIC   = "/rl/jump_signal"    # std_msgs/Int16

class HeightJumpRecorder(Node):
    def __init__(self):
        super().__init__("height_jump_recorder_node")

        # 最新值快照
        self.latest_height = None    # float
        self.latest_jump   = -1      # int（未收到前用 -1）
        self.has_height    = False
        self.has_jump      = False

        # 行缓冲：[time, height, jump_signal, gait_tag]
        self.rows = []

        # 计时起点
        self.start_time = self.get_clock().now()

        # 订阅
        self.height_sub = self.create_subscription(Float32, HEIGHT_TOPIC, self.height_cb, 10)
        self.jump_sub   = self.create_subscription(Int16,   JUMP_TOPIC,   self.jump_cb,   10)

        # 定时保存退出（第一次触发即保存）
        self.timer = self.create_timer(RECORD_DURATION, self.save_and_exit)

        self.get_logger().info(
            f"Height/Jump recorder started for {RECORD_DURATION:.2f}s. "
            f'Listening: "{HEIGHT_TOPIC}", "{JUMP_TOPIC}".'
        )
        self.get_logger().info(f"Output: {CSV_COMBINED_FILE}")

    # 相对起点的秒数
    def _now_seconds(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    # —— 回调：更新最新值，并立即写一行（带快照）——
    def height_cb(self, msg: Float32):
        try:
            self.latest_height = float(msg.data)
            self.has_height = True
        except Exception as e:
            self.get_logger().warn(f"Error parsing {HEIGHT_TOPIC}: {e}")
            return
        self._append_row(trigger="height")

    def jump_cb(self, msg: Int16):
        try:
            self.latest_jump = int(msg.data)
            self.has_jump = True
        except Exception as e:
            self.get_logger().warn(f"Error parsing {JUMP_TOPIC}: {e}")
            return
        self._append_row(trigger="jump")

    def _append_row(self, trigger: str):
        t = self._now_seconds()
        # 若尚未收到某一路数据，用占位（height=None → 空；jump=-1）
        height = self.latest_height if self.has_height else ""
        jump   = self.latest_jump if self.has_jump else -1

        self.rows.append([t, height, jump, GAIT_TAG])
        # 调试：查看写入频率/来源
        # self.get_logger().debug(f"append by {trigger} @ {t:.3f}s: h={height}, j={jump}")

    # —— 保存 CSV —— 
    def save_csv(self):
        if not self.rows:
            self.get_logger().info("No data recorded. Skip saving.")
            return

        header = ["Time (s)", "height", "jump_signal", "GaitTag"]
        with open(CSV_COMBINED_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)

        self.get_logger().info(f"Saved {len(self.rows)} rows to {CSV_COMBINED_FILE}")

    def save_and_exit(self):
        self.get_logger().info("Time up. Saving and shutting down...")
        self.save_csv()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HeightJumpRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted. Saving before exit...")
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
