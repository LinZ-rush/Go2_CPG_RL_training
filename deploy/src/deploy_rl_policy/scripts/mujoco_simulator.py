#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import mujoco.viewer
import mujoco
import numpy as np
#导入 Unitree 官方定义的 ROS 消息格式
from unitree_go.msg import LowState,LowCmd
from pathlib import Path
from std_msgs.msg import Float32MultiArray,Float32
import threading
import time
# 自动定位项目根目录，以便找到 go2.xml 模型文件
project_root=Path(__file__).parents[4]
class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator_node")

        # --- 核心通信接口 ---
        # 1. 发布机器人状态 (模仿真机底层驱动)
        # Topic: /mujoco/lowstate (包含关节角、IMU、速度等)
        self.low_state_puber=self.create_publisher(LowState,"/mujoco/lowstate",10)
        # 2. 调试用 Topic (可视化高度、速度、力)
        self.height_pub=self.create_publisher(Float32,"/mujoco/height",10)
        self.vel_pub=self.create_publisher(Float32MultiArray,"/mujoco/vel",10)
        self.force_pub=self.create_publisher(Float32MultiArray,"/mujoco/force",10)
        self.torque_pub=self.create_publisher(Float32MultiArray,"/mujoco/torque",10)
        # 3. 接收控制指令 (模仿真机接收底层指令)
        # Topic: /mujoco/lowcmd (RL 策略发出的动作)
        self.target_torque_suber=self.create_subscription(LowCmd,"/mujoco/lowcmd",self.target_torque_callback,10)
        
        self.step_counter = 0
        self.xml_path=project_root/"resources"/"go2"/"go2.xml"
        # Initialize Mujoco
        self.init_mujoco()
        self.target_dof_pos=[0]*12
        self.tau=[0.0]*12        
        # Load params
        self.timer = self.create_timer(0.001, self.publish_sensor_data)
        self.timer2=self.create_timer(0.001,self.update_tau)
        self.running=True
        self.kps=np.array([100.0]*12)
        self.kds=np.array([2.0]*12)
        self.recieve_data=False
        self.sim_thread=threading.Thread(target=self.step_simulation)
        self.sim_thread.start()

    def init_mujoco(self):
        """Initialize Mujoco model and data"""
        
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 0.001
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")
   
        
    def target_torque_callback(self,msg):
        self.recieve_data=True
        for i in range(12):
            self.target_dof_pos[i]=msg.motor_cmd[i].q
            self.kps[i]=msg.motor_cmd[i].kp
            self.kds[i]=msg.motor_cmd[i].kd
            
    def update_tau(self):
        if not self.recieve_data:
            return
        for i in range(12):
            self.tau[i]=self.pd_control(self.target_dof_pos[i],self.d.qpos[7+i],self.kps[i],self.d.qvel[6+i],self.kds[i])
        
    def step_simulation(self):
        while self.viewer.is_running() and self.running :
            if not self.recieve_data:
                continue
            step_start=time.perf_counter()
            """Main simulation step (executed in another thread)"""
            self.d.ctrl[:]=self.tau
            Torque=Float32MultiArray()
            Torque.data=self.tau
            self.torque_pub.publish(Torque)
            # Mujoco step
            mujoco.mj_step(self.m, self.d)  
            # Sync Mujoco viewer
            self.viewer.sync()
            time_until_next_step = self.m.opt.timestep - (time.perf_counter() - step_start)
            # print("time elapsed:",time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    def stop_simulation(self):
        self.running=False
        self.sim_thread.join()


    def publish_sensor_data(self):
        low_state_msg=LowState()
        for i in range(12):
            low_state_msg.motor_state[i].q=self.d.qpos[7+i]
            low_state_msg.motor_state[i].dq=self.d.qvel[6+i]
        low_state_msg.imu_state.quaternion=self.d.qpos[3:7].astype(np.float32)
        low_state_msg.imu_state.gyroscope=self.d.sensordata[40:43].astype(np.float32)
        self.low_state_puber.publish(low_state_msg)
        height_msg=Float32()
        height_msg.data=self.d.qpos[2]
        self.height_pub.publish(height_msg)
        vel=Float32MultiArray()
        vel.data=self.d.sensordata[52:55].astype(np.float32).tolist()
        self.vel_pub.publish(vel)
        # print("base vel:",self.d.sensordata[52:55])
        Force=Float32MultiArray()
        fl_site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "FL_foot_site")
        fl_site_xmat = np.reshape(self.d.site_xmat[fl_site_id], (3, 3))
        f1=self.d.sensordata[55:55+3]
        # print("raw force:",f1)
        f1=fl_site_xmat.dot(np.array(f1)).tolist()
        fr_site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "FR_foot_site")
        fr_site_xmat = np.reshape(self.d.site_xmat[fr_site_id], (3, 3))
        f2=self.d.sensordata[55+3:55+6]
        f2=fr_site_xmat.dot(np.array(f2)).tolist()
        rl_site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "RL_foot_site")
        rl_site_xmat = np.reshape(self.d.site_xmat[rl_site_id], (3, 3)) 
        f3=self.d.sensordata[55+6:55+9]
        f3=rl_site_xmat.dot(np.array(f3)).tolist()
        rr_site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "RR_foot_site")
        rr_site_xmat = np.reshape(self.d.site_xmat[rr_site_id], (3, 3))
        f4=self.d.sensordata[55+9:55+12]
        f4=rr_site_xmat.dot(np.array(f4)).tolist()
        Force.data=f1+f2+f3+f4
        # print("force:",f1[2],f2[2],f3[2],f4[2])
        self.force_pub.publish(Force)
    
    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        """Calculates torques from position commands"""
        torques=(target_q - q) * kp -  dq * kd
        return torques
    
def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_simulation()
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

