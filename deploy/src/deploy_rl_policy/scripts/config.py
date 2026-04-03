import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            # Low-Level Policy参数 (原有)
            self.policy_path = config["policy_path"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.ang_vel_scale = config["ang_vel_scale"]
            self.lin_vel_scale = config["lin_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            # High-Level Policy参数 (新增)
            self.high_policy_path = config["high_policy_path"]
            self.num_high_obs = config["num_high_obs"]
            self.num_high_actions = config["num_high_actions"]

            # 控制频率 (新增)
            self.high_decimation = config.get("high_decimation", 200)
            self.low_decimation = config.get("low_decimation", 10)

            # 决策融合参数 (新增)
            self.jump_feasibility_threshold = config.get("jump_feasibility_threshold", 0.7)
            self.enable_debug_output = config.get("enable_debug_output", False)
