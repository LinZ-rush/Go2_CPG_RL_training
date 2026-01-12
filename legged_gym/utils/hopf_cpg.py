import torch
import numpy as np

MU_LOW = 1
MU_UPP = 2

class HopfCPG:
    def __init__(self,
                 num_envs,
                 mu=1.0,  # intrinsic amplitude
                 alpha=50,
                 omega_swing=16* np.pi,
                 omega_stance=4 * np.pi,
                 coupling_strength=1.0,
                 couple=True,
                 time_step=0.005,
                 gait="TROT",  # TROT, WALK, BOUND, PACE
                 ground_clearance=0.07,
                 ground_penetration=0.01,
                 robot_height=0.3,
                 des_step_len=0.1,
                 max_step_len_rl=0.1,
                 mu_low = 1.0,
                 mu_up = 4.0,
                 use_RL=True,
                 device="cuda:0"):

        self.device = torch.device(device)
        self.num_envs = num_envs

        # Shape: (num_envs, 2, 4) — 2 for [r, theta]
        self.X = torch.zeros(num_envs, 2, 4, device=self.device)
        self.X_dot = torch.zeros_like(self.X)
        self.mu_low= mu_low,
        self.mu_up = mu_up,
        self._mu = mu
        self._alpha = alpha
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._ground_clearance = ground_clearance
        self._ground_penetration = ground_penetration
        self._robot_height = robot_height
        self._des_step_len = des_step_len
        self._max_step_len_rl = max_step_len_rl
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self.use_RL = use_RL
        self._set_gait(gait)
        if not self.use_RL:
            self.X[:, 0, :] = torch.rand((num_envs, 4), device=self.device) * 0.1
            # 使用PHI矩阵的第一行作为初始相位
            self.X[:, 1, :] = self.PHI[0, :].unsqueeze(0).repeat(num_envs, 1)
        else:
            self.X[:, 0, :] = MU_LOW
            # 随机初始相位 [0, 2π]
            self.X[:, 1, :] = torch.rand((num_envs, 4), device=self.device) * 2 * np.pi
        # self.X[:, 0, :]  = torch.rand((num_envs,4), device=self.device) * 0.1
        # self.X[:, 1, :] = torch.randn((num_envs, 4), device=self.device) * np.pi  # Random initial angles
        # self.X[:,1, :] = self.PHI[0, :]
        self._mu_rl= torch.full((num_envs, 4), MU_LOW, device=self.device)  # Initialize mu for R
        self._omega_rl = torch.zeros(num_envs, 4, device=self.device)




   
    def _set_gait(self, gait):
        """ Set coupling matrix PHI based on gait type (Torch version). """
        device = self.device
        # 定义相位耦合矩阵，每行为一个 oscillator 对其他 oscillator 的相位偏移
        if gait == "TROT":
            self.PHI = torch.tensor([
                [0, np.pi, np.pi, 0],
                [-np.pi, 0, 0, -np.pi],
                [-np.pi, 0, 0, -np.pi],
                [0, np.pi, np.pi, 0]
            ], device=device)

        elif gait == "WALK":
            self.PHI = torch.tensor([
                [0, np.pi, -np.pi / 2, np.pi / 2],
                [np.pi, 0, np.pi / 2, -np.pi / 2],
                [np.pi / 2, -np.pi / 2, 0, np.pi],
                [-np.pi / 2, np.pi / 2, np.pi, 0]
            ], device=device)

        elif gait == "BOUND":
            self.PHI = torch.tensor([
                [0, 0, np.pi, np.pi],
                [0, 0, np.pi, np.pi],
                [-np.pi, -np.pi, 0, 0],
                [-np.pi, -np.pi, 0, 0]
            ], device=device)

        elif gait == "PACE":
            self.PHI = torch.tensor([
                [0, np.pi, 0, np.pi],
                [-np.pi, 0, -np.pi, 0],
                [0, np.pi, 0, np.pi],
                [-np.pi, 0, -np.pi, 0]
            ], device=device)

        else:
            raise ValueError(f"Gait '{gait}' not implemented.")
    def update(self):
        if not self.use_RL:
            self._integrate_hopf_equations()
        else:
            self._integrate_hopf_equations_rl()
        if "OFFSETX" in self._rl_task_string:
            offset = self._scale_helper( a[:,8:12],-0.07, 0.07) 
            self.update_offset_x(offset)
        r = self.X[:, 0, :]
        theta = self.X[:, 1, :]
        z = torch.where(
            torch.sin(theta) > 0,
            -self._robot_height + self._ground_clearance * torch.sin(theta),
            -self._robot_height + self._ground_penetration * torch.sin(theta)
        )
        if not self.use_RL:
            # For RL, we use a fixed step length based on the desired step length
            x = -self._des_step_len * r * torch.cos(theta)
        else:
            r_clipped = torch.clamp(r, MU_LOW, MU_UPP)
            x = -self._max_step_len_rl * (r_clipped - MU_LOW) * torch.cos(self.X[:, 1, :])
        return x, z

    def _integrate_hopf_equations(self):
        """Torch-based Hopf oscillator with coupling — batched over num_envs."""
        X = self.X.clone()  # shape: [num_envs, 2, 4]
        X_dot_prev = self.X_dot.clone()
        X_dot = torch.zeros_like(self.X)

        r = X[:, 0, :]  # shape: [num_envs, 4]
        theta = X[:, 1, :]  # shape: [num_envs, 4]

        # Compute r_dot: α(μ - r²)r
        r_dot = self._alpha * (self._mu - r ** 2) * r  # shape: [num_envs, 4]

        # Determine omega: swing if θ ∈ [0, π), else stance
        omega = torch.where(
            (theta >= 0) & (theta < np.pi),
            torch.tensor(self._omega_swing, device=self.device),
            torch.tensor(self._omega_stance, device=self.device)
        )  # shape: [num_envs, 4]


        if self._couple:
            # 创建相位差矩阵 [num_envs, 4, 4]
            theta_i = theta.unsqueeze(2)  # [num_envs, 4, 1]
            theta_j = theta.unsqueeze(1)  # [num_envs, 1, 4]
            theta_diff = theta_j - theta_i  # θ_j - θ_i [num_envs, 4, 4]
            
            PHI = self.PHI.unsqueeze(0)  # [1, 4, 4]
            coupling_terms = self._coupling_strength * r.unsqueeze(1) * torch.sin(theta_diff - PHI)
            
            mask = torch.eye(4, device=self.device).bool().unsqueeze(0)
            coupling_terms = coupling_terms.masked_fill(mask, 0)
            
            coupling_sum = coupling_terms.sum(dim=2)
            omega += coupling_sum
        # Assemble derivatives
        X_dot[:, 0, :] = r_dot
        X_dot[:, 1, :] = omega

        # Integrate with semi-implicit Euler
        self.X = X + 0.5 * self._dt * (X_dot + X_dot_prev)
        self.X_dot = X_dot
        self.X[:, 1, :] = self.X[:, 1, :] % (2 * np.pi)  # keep θ ∈ [0, 2π)
        # print(f"THETA: {self.X[:, 1, :]}")


    def _integrate_hopf_equations_rl(self):
        X = self.X.clone()
        X_dot_prev = self.X_dot.clone()
        X_dot = torch.zeros_like(self.X)

        r = X[:, 0, :]
        r_dot = self._alpha * (self._mu_rl - r**2) * r
        theta_dot = self._omega_rl

        X_dot[:, 0, :] = r_dot
        X_dot[:, 1, :] = theta_dot

        self.X = X + 0.5 * self._dt * (X_dot + X_dot_prev)
        self.X_dot = X_dot
        self.X[:, 1, :] = self.X[:, 1, :] % (2 * np.pi)

    def get_r(self):
        return self.X[:, 0, :]

    def get_theta(self):
        return self.X[:, 1, :]

    def get_dr(self):
        return self.X_dot[:, 0, :]

    def get_dtheta(self):
        return self.X_dot[:, 1, :]

    def set_omega_rl(self, omegas):
        omegas= self._scale_helper(omegas, 0, 9*np.pi)
        # print(f"Setting omegas: {omegas}")
        self._omega_rl = omegas.to(self.device)

    def set_mu_rl(self, mus):
        mus=self._scale_helper(mus, MU_LOW, MU_UPP*2)
        # print(f"Setting mus: {mus}")
        self._mu_rl = mus.to(self.device)

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Scale from [-1, 1] to [lower_lim, upper_lim] using torch."""
        new_a = lower_lim + 0.005 * (action + 100) * (upper_lim - lower_lim)
        return torch.clamp(new_a, lower_lim, upper_lim)

    def reset(self,env_ids):
        if len(env_ids)==0:
            return 
        # Reset the CPG state for the specified environments
        if not self.use_RL:
            self.X[env_ids, 0, :]  = torch.rand((len(env_ids),4), device=self.device) * 0.1
            self.X[env_ids, 1, :] = self.PHI[0,:] # Random initial angles
        else:
            self.X[env_ids, 0, :] = MU_LOW
            self.X[env_ids, 1, :] = torch.rand((len(env_ids), 4), device=self.device)*np.pi
        self.X_dot[env_ids] = 0.0
        # self._mu_rl[env_ids] = MU_LOW
        # self._omega_rl[env_ids] = 0.0