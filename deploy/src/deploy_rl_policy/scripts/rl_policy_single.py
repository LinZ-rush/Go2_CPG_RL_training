#!/usr/bin/env python3
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
from xbox_command import XboxController
from std_msgs.msg import Float32MultiArray,Int16,Float32
from collections import deque
import time
class asset:
    hip_link_length_go2 = 0.0955
    thigh_link_length_go2 = 0.213
    calf_link_length_go2 = 0.213    
    
project_root=Path(__file__).parents[4]
class dataReciever(Node):
    def __init__(self,config:Config):
        super().__init__("policy_inference_node")
        self.config = config
        self.cmd_sub=XboxController(self)
        self._cpg = CPG_RL(time_step=0.001,num_envs=1,rl_task_string="CPG_OFFSETX")
        # Initialize the policy network()
        script_dir = os.path.dirname(os.path.abspath(__file__))  
        policy_path = os.path.join(script_dir, self.config.policy_path)
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.cur_obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.low_state=LowState()
        self.contact_deque = deque(maxlen=30)
        self.count=0
        self.force_contact=True
        self.landing=True
        self.jump_signal=0
        self.prev_jump_signal=0
        self.prev_jump_pressed=0
        self.mode_timer=0
        self.gait_idx=0
        self.gait_pressed=0
        self.prev_gait_pressed=0
        self.jump_factor=1.0
        self.step_start=time.perf_counter()
        if args.simulation:
            self.low_state_sub=self.create_subscription(LowState,"/mujoco/lowstate",self.low_state_callback,10)
            print("reading data from simuation")
        else:    
            self.low_state_sub=self.create_subscription(LowState,"/lowstate",self.low_state_callback,10) #500 HZ
            print("reading data from reality")


        self.get_logger().info("Waiting for data")
        self.timer = self.create_timer(0.001, self.run)
        self.gait_id_puber=self.create_publisher(Int16,"/rl/gait_id",10)
        self.force_sub=self.create_subscription(Float32MultiArray,"/mujoco/force",self.force_callback,10)
        self.target_vel_puber=self.create_publisher(Float32MultiArray,"/rl/target_vel",10)
        self.target_pos_puber=self.create_publisher(Float32MultiArray,"/rl/target_pos",10)
        self.jump_sig_puber=self.create_publisher(Int16,"/rl/jump_signal",10)
    def low_state_callback(self,msg:LowState):
        self.low_state=msg
        
    def force_callback(self,msg:Float32MultiArray):
        if len(msg.data) >= 12:
            fl_force_z = msg.data[2]   # FL 脚 Z 轴力
            fr_force_z = msg.data[5]   # FR 脚 Z 轴力
            rl_force_z = msg.data[8]   # RL 脚 Z 轴力
            rr_force_z = msg.data[11]  # RR 脚 Z 轴力
        else:
            self.get_logger().warn('Received force message with insufficient data. Skipping.')
            return

        # 判断是否接触地面，转换为二进制 (1/0)
        fl_contact = 1 if fl_force_z <- 30 else 0
        fr_contact = 1 if fr_force_z <- 30 else 0
        rl_contact = 1 if rl_force_z <- 30 else 0
        rr_contact = 1 if rr_force_z <- 30 else 0
        self.force_contact=fl_contact+fr_contact+rl_contact+rr_contact>=1
        self.landing=fl_contact+fr_contact+rl_contact+rr_contact>=4

    def run(self):
        # self.get_logger().info("running")
        # Get the current joint position and velocity
        # print(time.perf_counter()-self.step_start)
        # self.step_start=time.perf_counter()
        if (self.cmd_sub.axes and self.cmd_sub.axes[2] == -1 and self.cmd_sub.axes[5] == -1):
            sys.exit()
            
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # comment these two lines for simulation
        self.qj=self.qj[sequence]
        self.dqj=self.dqj[sequence]
        
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        # create observation
        gravity_orientation = self.get_gravity_orientation(quat)
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale

        ang_vel = ang_vel * self.config.ang_vel_scale
        self.left_button,self.right_button=self.cmd_sub.is_pressed()
        if self.left_button and self.right_button:
            if self.cmd_sub.axes[7]==0:
                self.cmd_sub.linear_x+=-np.sign(self.cmd_sub.linear_x)*0.001
            else:
                self.cmd_sub.linear_x+=np.sign(self.cmd_sub.axes[7])*0.002
            if self.cmd_sub.axes[6]==0:
                self.cmd_sub.linear_y+=-np.sign(self.cmd_sub.linear_y)*0.001
            else:
                self.cmd_sub.linear_y+=np.sign(self.cmd_sub.axes[6])*0.002
            self.cmd_sub.linear_x=np.clip(self.cmd_sub.linear_x,-self.cmd_sub.max_speed,self.cmd_sub.max_speed)
            self.cmd_sub.linear_y=np.clip(self.cmd_sub.linear_y,-self.cmd_sub.max_speed,self.cmd_sub.max_speed)
            self.cmd_sub.angular_z=self.cmd_sub.get_right_stick()
        else:
            self.cmd_sub.linear_x+=-np.sign(self.cmd_sub.linear_x)*0.002
            self.cmd_sub.linear_y+=-np.sign(self.cmd_sub.linear_y)*0.002
            self.cmd_sub.angular_z+=-np.sign(self.cmd_sub.angular_z)*0.002
        self.cmd=np.array([self.cmd_sub.linear_x,self.cmd_sub.linear_y,self.cmd_sub.angular_z])
        current_pressed = self.cmd_sub.jump_pressed()
        current_contact = self.force_contact
        self.contact_deque.append(current_contact)
        # stable_contact = sum(self.contact_deque)==30
        # base_stable = np.linalg.norm(gravity_orientation[:2]) < 0.1
        if current_contact and current_pressed and not self.prev_jump_pressed :
            self.jump_signal = 1
            self.mode_timer=0
        # print("current_contact:",current_contact)
        if current_contact and self.prev_jump_signal and self.mode_timer>500:
            self.jump_signal = 0
            self.jump_factor=0.5

        self.gait_pressed=self.cmd_sub.gait_pressed()
        if self.gait_pressed and not self.prev_gait_pressed:
            self.gait_idx+=1
            if self.gait_idx>3:
                self.gait_idx=0
        gait_msg=Int16()
        gait_msg.data=self.gait_idx
        self.gait_id_puber.publish(gait_msg)
        target_height=0.55 if self.jump_signal else 0.3
        if self.jump_signal: 
            self.jump_factor=1.3 
        gait_factor=1.0 if (self.gait_idx==0 or self.gait_idx==2) else 1.0
        vel_cmd=self.cmd*self.jump_factor*gait_factor
        vel_cmd=np.clip(vel_cmd,0,2.2)
        # vel_cmd=np.array([1.0,0.0,0.0])
        vel_msg=Float32MultiArray()
        vel_msg.data=vel_cmd.tolist()
        self.target_vel_puber.publish(vel_msg)
        if self.count%10==0:    
            # print(self.cmd)
            self.cur_obs[:3] = ang_vel
            self.cur_obs[3:6] = gravity_orientation
            self.cur_obs[6:9] = vel_cmd * self.config.cmd_scale
            self.cur_obs[9:10]=target_height  #jump height
            self.cur_obs[10:11]=self.gait_idx  #gait command
            self.cur_obs[11:12]=self.jump_signal   #jump signal
            self.cur_obs[12: 24] = qj_obs
            self.cur_obs[24:36] = dqj_obs
            self.cur_obs[36:52] = self.action
            cpg_r=(self._cpg.X[:,0,:] - ((self._cpg.mu_up[0]+ self._cpg.mu_low[0]) / 2)) * self.config.dof_pos_scale
            self.cur_obs[52:56] = cpg_r
            cpg_theta=(self._cpg.X[:,1,:] - np.pi) * 1/np.pi
            self.cur_obs[56:60] = cpg_theta
            cpg_r_dot= self._cpg.X_dot[:,0,:] * 1/30
            self.cur_obs[60:64] = cpg_r_dot
            cpg_theta_dot=(self._cpg.X_dot[:,1,:] - 15) * 1/30
            self.cur_obs[64:68] = cpg_theta_dot
            self.cur_obs=np.clip(self.cur_obs,-100,100)
            # Get the action from the policy network
            obs_tensor = torch.from_numpy(self.cur_obs).unsqueeze(0)
            self.action = self.policy(obs_tensor).detach()
            self.action = torch.clip(self.action, -100.0, 100.0)
        # transform action to target_dof_pos
        actions_scaled = self.action * self.config.action_scale
        des_joint_pos = torch.zeros(1, 12)
        xs,ys,zs=self._cpg.get_CPG_RL_actions(actions_scaled, 31.4, -3.14)
        sideSign = np.array([-1,1,-1,1]) 
        foot_y = torch.ones(1,requires_grad=False) * asset.hip_link_length_go2
        LEG_INDICES = np.array([1,0,3,2])
        if self.count%10==0:
            for ig_idx,i in enumerate(LEG_INDICES):
                x = xs[:,i]
                z = zs[:,i]
                y = sideSign[i] * foot_y  + ys[:,i]
                des_joint_pos[:, 3*i:3*i+3] = self._cpg.compute_inverse_kinematics(asset,i,x,y,z)
                # print("x,z:",x.item(),z.item())
            self.target_dof_pos = des_joint_pos.numpy().squeeze()
            #tau computed by network in the sequence of: 
            ''' 
            FL_hip,FL_thigh,FL_calf
            FR_hip,FR_thigh,FR_calf
            RL_hip,RL_thigh,RL_calf
            RR_hip,RR_thigh,RR_calf
            '''
            # The true order of the actuator in /lowcmd (also in mujoco simulator) should be:
            ''' 
            FR_hip,FR_thigh,FR_calf
            FL_hip,FL_thigh,FL_calf
            RR_hip,RR_thigh,RR_calf
            RL_hip,RL_thigh,RL_calf
            '''
            msg=Float32MultiArray()
            msg.data.extend(self.target_dof_pos.astype(np.float32)[sequence].tolist())
            self.target_pos_puber.publish(msg)
        self.count+=1
        jump_sig_msg=Int16()
        jump_sig_msg.data=self.jump_signal
        self.jump_sig_puber.publish(jump_sig_msg)
        print("jump_signal:",self.jump_signal)
        self.prev_jump_signal = self.jump_signal
        self.prev_jump_pressed=current_pressed
        self.prev_gait_pressed=self.gait_pressed
        self.mode_timer+=1
        # print("time elapsed in one step:",time.perf_counter() - step_start)


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
    reciever_node=dataReciever(config=config)
    rclpy.spin(reciever_node)
    reciever_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    # Load config
    config_path = project_root/"src"/"deploy_rl_policy"/"configs"/"go2.yaml"
    config = Config(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_simulation', type=str, choices=["True", "False"], default="True")
    args = parser.parse_args()
    args.simulation = args.is_simulation == "True"
    main()