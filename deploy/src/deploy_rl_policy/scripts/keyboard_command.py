#!/usr/bin/env python3
"""
键盘控制器 - 替代 XboxController + Joy Node

控制方式:
  状态控制 (对应 low_level_ctrl.cpp 状态机):
    1:       站起 (Standing Up)     — 对应Xbox B键
    2:       趴下 (Laying Down)     — 对应Xbox A键
    3:       执行策略 (Run Policy)  — 对应Xbox LB+RB

  运动控制 (策略执行后生效):
    ↑ / ↓    前进/后退 (长按逐渐加速)
    ← / →    左移/右移
    Q:       Walk 步态演示 (速度 0.3 m/s)
    W:       Trot 步态演示 (速度 1.0 m/s)
    E:       Bound 步态演示 (速度 1.8 m/s)
    R:       转向 (右旋)
    空格:    跳跃
    S:       急停
    ESC:     退出

使用流程: 按1站起 → 按3执行策略 → 方向键/QWE控制运动 → 空格跳跃
"""
import sys
import tty
import termios
import select
import threading
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy


# 步态演示速度 (对应训练中的步态-速度区间)
GAIT_DEMO_SPEEDS = {
    'q': 0.3,   # Walk: speed < 0.5
    'w': 1.0,   # Trot: 0.5 <= speed < 1.5
    'e': 1.8,   # Bound: speed >= 1.5
}

# 加速/减速参数
ACCEL_STEP = 0.05       # 每次按键加速量 (m/s)
DECAY_RATE = 0.005      # 无输入时自然减速量 (m/s per tick)
MAX_SPEED = 2.0         # 最大速度 (m/s)
ANGULAR_STEP = 0.05     # 转向加速量 (rad/s)
MAX_ANGULAR = 1.0       # 最大角速度 (rad/s)


class KeyboardController:
    def __init__(self, node: Node):
        self.node = node
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        self.max_speed = MAX_SPEED

        # 跳跃状态
        self._jump_flag = False
        self._jump_consumed = True

        # 退出标志
        self._exit_requested = False

        # 上一次收到按键的时间 (用于自然减速)
        self._last_key_time = time.time()

        # Joy消息发布器 (供 low_level_ctrl.cpp 状态机使用)
        self._joy_pub = node.create_publisher(Joy, '/joy', 10)
        # buttons布局: [A(0), B(1), X(2), Y(3), LB(4), RB(5), ...]
        self._joy_buttons = [0] * 11
        self._joy_axes = [0.0] * 8

        # 终端原始设置 (用于恢复)
        self._old_settings = termios.tcgetattr(sys.stdin)

        # 启动键盘监听线程
        self._running = True
        self._key_thread = threading.Thread(target=self._key_listener, daemon=True)
        self._key_thread.start()

        # 定时发布Joy消息 (20Hz, 让 low_level_ctrl 持续收到)
        self._joy_timer = node.create_timer(0.05, self._publish_joy)

        node.get_logger().info(
            "Keyboard control enabled:\n"
            "  1: Stand up | 2: Lay down | 3: Run policy\n"
            "  Arrow keys: move (long press to accelerate)\n"
            "  Q/W/E: Walk(0.3)/Trot(1.0)/Bound(1.8) demo speed\n"
            "  Space: jump | S: stop | R: rotate | ESC: exit\n"
            "  Flow: press 1 -> press 3 -> arrow keys to move"
        )

    def _key_listener(self):
        """后台线程: 监听键盘输入"""
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
                else:
                    self._apply_decay()
        except Exception as e:
            print(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _handle_key(self, ch):
        """处理按键"""
        self._last_key_time = time.time()

        # 先清除所有按钮 (单次触发)
        self._joy_buttons = [0] * 11

        if ch == '\x1b':
            # 转义序列 (方向键)
            seq = sys.stdin.read(2)
            if seq == '[A':  # ↑ 前进
                self.linear_x = np.clip(self.linear_x + ACCEL_STEP, -MAX_SPEED, MAX_SPEED)
            elif seq == '[B':  # ↓ 后退
                self.linear_x = np.clip(self.linear_x - ACCEL_STEP, -MAX_SPEED, MAX_SPEED)
            elif seq == '[C':  # → 右移
                self.linear_y = np.clip(self.linear_y - ACCEL_STEP, -MAX_SPEED, MAX_SPEED)
            elif seq == '[D':  # ← 左移
                self.linear_y = np.clip(self.linear_y + ACCEL_STEP, -MAX_SPEED, MAX_SPEED)
            elif seq == '' or seq[0] == '':
                self._exit_requested = True

        elif ch == '1':
            # 站起 → Joy buttons[1] = B
            self._joy_buttons[1] = 1
            self.node.get_logger().info(">> STAND UP")

        elif ch == '2':
            # 趴下 → Joy buttons[0] = A
            self._joy_buttons[0] = 1
            self.node.get_logger().info(">> LAY DOWN")

        elif ch == '3':
            # 执行策略 → Joy buttons[4]=LB + buttons[5]=RB
            self._joy_buttons[4] = 1
            self._joy_buttons[5] = 1
            self.node.get_logger().info(">> RUN POLICY")

        elif ch == ' ':
            self._jump_flag = True
            self._jump_consumed = False

        elif ch.lower() in GAIT_DEMO_SPEEDS:
            target_speed = GAIT_DEMO_SPEEDS[ch.lower()]
            self.linear_x = target_speed
            self.linear_y = 0.0
            self.angular_z = 0.0

        elif ch.lower() == 'r':
            self.angular_z = np.clip(self.angular_z - ANGULAR_STEP, -MAX_ANGULAR, MAX_ANGULAR)

        elif ch.lower() == 's':
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.angular_z = 0.0

        elif ch == '\x03':
            self._exit_requested = True

    def _apply_decay(self):
        """无输入时自然减速"""
        elapsed = time.time() - self._last_key_time
        if elapsed > 0.2:
            if abs(self.linear_x) > DECAY_RATE:
                self.linear_x -= np.sign(self.linear_x) * DECAY_RATE
            else:
                self.linear_x = 0.0

            if abs(self.linear_y) > DECAY_RATE:
                self.linear_y -= np.sign(self.linear_y) * DECAY_RATE
            else:
                self.linear_y = 0.0

            if abs(self.angular_z) > DECAY_RATE:
                self.angular_z -= np.sign(self.angular_z) * DECAY_RATE
            else:
                self.angular_z = 0.0

    def _publish_joy(self):
        """定时发布Joy消息到 /joy topic (供 low_level_ctrl.cpp 使用)"""
        msg = Joy()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.axes = self._joy_axes
        msg.buttons = self._joy_buttons
        self._joy_pub.publish(msg)
        # 按钮发布一次后清除 (模拟单次按下)
        self._joy_buttons = [0] * 11

    # ===== 与 XboxController 兼容的接口 =====

    def jump_pressed(self):
        """返回跳跃信号 (一次性, 读取后清除)"""
        if self._jump_flag and not self._jump_consumed:
            self._jump_consumed = True
            return 1
        self._jump_flag = False
        return 0

    def is_pressed(self):
        """兼容接口: 键盘模式下始终返回 (True, True) 表示控制激活"""
        return True, True

    def is_exit_requested(self):
        """检查是否请求退出"""
        return self._exit_requested

    def stop(self):
        """停止键盘监听, 恢复终端设置"""
        self._running = False
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass
