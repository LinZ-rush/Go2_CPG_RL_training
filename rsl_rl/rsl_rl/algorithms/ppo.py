# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        # 1. 暂存奖励和结束信号
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        # 如果环境是因为“时间到了”(max_episode_length) 而结束，而不是因为“摔倒了”而结束：
        if 'time_outs' in infos:
            # 我们必须把下一时刻的预期价值 (Value) 加回到当前的 Reward, 因为 PPO 的 return 计算是基于 TD(lambda) 的，需要一个“未来价值”的估计。
            # 如果不加上这部分，PPO 会误以为“时间到了”这个状态的 Value 是 0，这会导致它刻意避免存活太久，或者在最后几步胡乱行动
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition) # 把在 act() 里收集的 obs, action 和这里的 rewards, dones 打包塞进显存
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)


    def update(self):
        """
        PPO 算法的核心更新步骤。
        利用 RolloutStorage 中收集到的数据 (Trajectory)，对 Actor 和 Critic 网络进行多轮次 (Epochs) 的梯度更新。
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0

        # 1. 准备数据生成器 (Mini-batch Generator)
        # 这一步将巨大的 Buffer 数据打乱，并切分成小批次 (Mini-batch)
        if self.actor_critic.is_recurrent:
            # 如果是 RNN 网络 (LSTM/GRU)，需要特殊的生成器来保证时序数据的连续性
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # 标准 MLP 网络使用普通生成器
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # 2. 开始多轮次 (Epoch) 遍历
        # 这里的循环会执行 num_learning_epochs * num_mini_batches 次
        # obs_batch: 观测数据
        # target_values_batch: 真实回报 (Return = GAE + Value)
        # advantages_batch: 优势函数 (Advantage)，用于衡量动作比平均水平好多少
        # old_actions_log_prob_batch: 收集数据时的旧策略概率 (用于计算 Ratio）
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            
                # 3. 重新评估 (Re-evaluation)
                # 使用当前最新的网络参数，重新计算这批数据的统计量
                # 前向传播 Actor，得到新的动作分布参数 (mu, sigma)
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                # 计算新策略下的动作对数概率: log(pi_new(a|s))
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # 前向传播 Critic，得到新的状态价值预测: V_new(s)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                # 获取当前分布的均值、标准差和熵
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # 4. 基于 KL 散度的自适应学习率 (Adaptive Learning Rate based on KL)
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        # 计算两个高斯分布之间的 KL 散度公式: KL(N_old || N_new)
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        # 如果变化太剧烈 (KL > 目标值的2倍)，说明步子迈大了，学习率除以 1.5 (刹车)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        # 如果变化太微小 (KL < 目标值的1/2)，说明太保守了，学习率乘以 1.5 (加油)    
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        # 将新的学习率应用到优化器中
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # 5. 计算策略损失 (Surrogate Loss)
                # 计算概率比率 Ratio = exp(log_new - log_old) = pi_new / pi_old
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # Loss1 = -Advantage * Ratio (未截断的普通策略梯度)
                surrogate = -torch.squeeze(advantages_batch) * ratio
                # Loss2 = -Advantage * Clip(Ratio, 1-eps, 1+eps) (截断后的保守梯度)
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                # 取两者的最大值 (因为带了负号，相当于取原始公式的最小值 Min)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # 6. 计算价值损失 (Value Function Loss)
                # 我们希望 Critic 的预测值 V(s) 尽可能接近真实回报 (returns_batch)
                if self.use_clipped_value_loss:
                    # Value Clip （可选）: 如果新预测的 V 值偏离旧 V 值太远，也把它截断。
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    # 标准 MSE Loss
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # 7. 总损失 (Total Loss)
                # Total = Policy Loss + c1 * Value Loss - c2 * Entropy
                # 减去 Entropy 是为了最大化熵 (鼓励探索)，因为这里是在做 minimize loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # 8. 梯度更新 (Gradient Step)
                self.optimizer.zero_grad()    # 清空旧梯度
                loss.backward()               # 反向传播计算新梯度
                # 梯度裁剪 (Gradient Clipping): 防止梯度爆炸，稳定训练
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()         # 更新网络参数

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
        # 计算平均损失并清理 Storage
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
