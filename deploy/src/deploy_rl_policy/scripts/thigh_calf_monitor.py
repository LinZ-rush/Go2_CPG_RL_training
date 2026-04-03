#!/usr/bin/env python3
"""
独立小工具: 订阅 /mujoco/lowstate, 提取四肢 thigh 和 calf 关节角度/速度并打印。
无需编译, 直接运行:
    python3 thigh_calf_monitor.py
(需要先 source ros2 和 unitree_go 的 setup)
"""

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState
import numpy as np

# motor_state 索引: FR(0,1,2) FL(3,4,5) RR(6,7,8) RL(9,10,11)
# hip=0,3,6,9  thigh=1,4,7,10  calf=2,5,8,11
JOINT_MAP = {
    "FR_thigh": 1, "FR_calf": 2,
    "FL_thigh": 4, "FL_calf": 5,
    "RR_thigh": 7, "RR_calf": 8,
    "RL_thigh": 10, "RL_calf": 11,
}

class ThighCalfMonitor(Node):
    def __init__(self):
        super().__init__("thigh_calf_monitor")
        self.sub = self.create_subscription(
            LowState, "/mujoco/lowstate", self.callback, 10
        )
        self.count = 0
        self.get_logger().info("Thigh/Calf monitor started, subscribing /mujoco/lowstate ...")

    def callback(self, msg):
        self.count += 1
        # 每 100 次打印一次 (~10Hz), 避免刷屏
        if self.count % 100 != 0:
            return

        lines = ["\n===== Thigh & Calf Joints ====="]
        for name, idx in JOINT_MAP.items():
            pos = msg.motor_state[idx].q
            vel = msg.motor_state[idx].dq
            lines.append(f"  {name:10s}  pos={pos:+7.3f} rad  vel={vel:+7.3f} rad/s")
        lines.append("=" * 32)
        self.get_logger().info("\n".join(lines))

def main():
    rclpy.init()
    node = ThighCalfMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
