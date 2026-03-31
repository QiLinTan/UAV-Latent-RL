                                            # =============================================================
# TD3-Latent (UAV Hierarchical Control Version): 
# 无人机层级异步控制算法
# -------------------------------------------------------------
# 核心改进：
#   1) 状态语义拆分：s_high（任务级） + s_low（执行级）
#   2) Burn-in 序列训练：让 GRU 记忆在训练中真正生效
#   3) 动作低通滤波：Decoder 后的一阶滤波（只用于执行，Critic 评估原始动作）
#   4) 频率适配：Encoder 1/5~1/10 Decoder 频率的语义异步控制
#
# 设计哲学：
#   - Encoder + GRU：外环状态生成器
#   - latent：外环控制意图
#   - Decoder：学习型内环执行器
#   - Low-pass filter：执行器物理约束
# =============================================================

import os
os.environ['MUJOCO_GL'] = 'glfw'
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
except ImportError:
    print("Warning: gym_pybullet_drones not found. Please install it to run this code.")
    HoverAviary, ObservationType, ActionType = None, None, None


# =============================================================
# 工具函数：多层感知机（MLP）
# =============================================================

def mlp(input_dim: int, hidden_dim: int, output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """构建一个 3 层的前馈神经网络（MLP）"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    )


# =============================================================
# 状态拆分工具函数（无人机场景）
# =============================================================

def split_state_uav(state: np.ndarray, state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将状态拆分为 s_high (历史序列) 和 s_low (当前状态)
    适配无人机 72 维混合观察空间:
    - s_low (12维): 当前运动学状态 [x, y, z, r, p, y, vx, vy, vz, wx, wy, wz]
    - s_high (60维): 15 步历史动作序列 [a_t-15, ..., a_t-1]

    注意：为了适配现有代码流 (s_high -> Encoder, s_low -> Decoder)，
    我们将历史序列命名为 s_high，当前状态命名为 s_low。
    """
     # 🔴 添加调试信息
    print(f"split_state_uav 被调用，state shape: {state.shape}, state_dim: {state_dim}")
    
    # 如果状态已经是12维，返回空s_high和完整state作为s_low
    if len(state) == 12:
        return np.array([]), state  # s_high为空，s_low为完整状态

    # s_low: 12-dim kinematics
    s_low = state[:12]
    # s_high: 60-dim action history (15 steps * 4 dims)
    s_high = state[12:]
    return s_high, s_low


# =============================================================
# ActorEncoder：GRU 版本，s_high -> latent z
# (修改后：输入为历史动作序列)
# =============================================================

class ActorEncoder(nn.Module):
    """
    历史序列编码器 (GRU 版本)
        输入：s_high (历史动作序列)
        输出：latent z (从历史中提取的特征)
    
    关键特性：
    - 使用 GRU 捕捉时序依赖
    - 只接收 s_high，不接收完整状态
    """

    def __init__(self, high_state_dim: int, latent_dim: int, gru_hidden_dim: int = 256):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.latent_dim = latent_dim
        
        # 假设 high_state_dim 是历史动作序列的平铺维度 (e.g., 60)
        # 假设每个动作是 4 维
        self.action_dim_per_step = 4
        self.seq_len = high_state_dim // self.action_dim_per_step

        # GRU 层：输入 s_high，输出 hidden state
        self.gru = nn.GRU(
            input_size=self.action_dim_per_step, # 4
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # 从 GRU hidden state 映射到 latent code
        self.latent_proj = nn.Linear(gru_hidden_dim, latent_dim)

    def init_hidden(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """初始化 hidden state"""
        h = torch.zeros(1, batch_size, self.gru_hidden_dim)
        if device is not None:
            h = h.to(device)
        return h

    def forward(self, s_high: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            s_high: [batch_size, high_state_dim] (e.g., 60)
            hidden: [1, batch_size, gru_hidden_dim]（可选）
        输出:
            latent z: [batch_size, latent_dim]
            hidden: [1, batch_size, gru_hidden_dim]
        """
        batch_size = s_high.shape[0]

        # Reshape from (batch, 60) to (batch, 15, 4)
        s_high_seq = s_high.view(batch_size, self.seq_len, self.action_dim_per_step)

        # 如果没有提供 hidden，初始化为零
        if hidden is None:
            device = s_high_seq.device if hasattr(s_high_seq, 'device') else torch.device('cpu')
            hidden = self.init_hidden(s_high_seq.shape[0], device=device)
        
        # GRU 前向传播
        gru_out, hidden_next = self.gru(s_high_seq, hidden)
        
        # 取最后一个时间步的输出
        if gru_out.dim() == 3:
            gru_out = gru_out[:, -1, :]  # [batch_size, gru_hidden_dim]
        
        # 映射到 latent code
        latent = self.latent_proj(gru_out)  # [batch_size, latent_dim]
        
        return latent, hidden_next


# =============================================================
# Decoder：内环执行器 (z, s_low) -> action
# =============================================================

class ActorDecoder(nn.Module):
    """
    低层策略（解码器）：内环执行器
        输入：(z, s_low)
        输出：raw action
    
    关键约束：
    - 只接收 s_low，不能看到完整的 s_high
    - 这是真正的"内环"，在语义上受外环 latent 约束
    """

    def __init__(
        self,
        low_state_dim: int,
        action_dim: int,
        latent_dim: int,
        action_max: float,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.action_max = action_max
        # 输入维度 = latent_dim + low_state_dim (当前运动学状态)
        self.net = mlp(latent_dim + low_state_dim, hidden_dim, action_dim)

    def forward(self, latent: torch.Tensor, s_low: torch.Tensor) -> torch.Tensor:
        """
        输入:
            latent: [batch_size, latent_dim]
            s_low:  [batch_size, low_state_dim]
        输出:
            raw_action: [batch_size, action_dim]（未滤波的原始动作）
        """

        # 🔴 添加断言
        assert s_low.shape[-1] == 12, f"s_low 维度错误: {s_low.shape[-1]}，应该是12"

        # 拼接 latent 和 s_low (当前运动学状态)
        x = torch.cat([latent, s_low], dim=-1)
        # tanh 用于约束动作范围，再缩放到环境动作上限
        raw_action = torch.tanh(self.net(x)) * self.action_max
        return raw_action


# =============================================================
# Critic 网络（Q(s, a)）：评估完整状态和动作
# =============================================================

class Critic(nn.Module):
    """
    TD3 中的 Q 网络：
        输入 (s, a_raw)，输出标量 Q 值
    
    重要：Critic 评估的是完整的 s 和 a_raw（未滤波的动作）
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q = mlp(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
         # 🔴 添加断言
        assert state.shape[-1] == 12, f"Critic state 维度错误: {state.shape[-1]}，应该是12"
        assert action.shape[-1] == 4, f"Critic action 维度错误: {action.shape[-1]}，应该是4"

        x = torch.cat([state, action], dim=-1)
        return self.q(x)


# =============================================================
# 简化的 Forward Dynamics Model
# 输入: (state, action) -> 预测 next s_low (运动学子向量)
# =============================================================


class DynamicsModel(nn.Module):
    """
    简化的确定性前向动力学模型：
        输入: concat(state, action)
        输出: 预测的 next_s_low (用于 dynamics loss)
    只预测 low-level 运动学（例如 12-dim），避免过度拟合完整高维观测。
    """

    def __init__(self, low_state_dim: int, action_dim: int, pred_low_dim: int, hidden_dim: int = 512):
        super().__init__()
        input_dim = low_state_dim + action_dim
        self.net = mlp(input_dim, hidden_dim, pred_low_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be already concatenated input vector
        return self.net(x)


# =============================================================
# Replay Buffer：支持序列采样（Burn-in 训练）
# =============================================================

class SequenceReplayBuffer:
    """
    支持序列采样的经验回放池
    
    关键特性：
    - 存储完整的 (s, a_raw, r, s', done) transitions
    - 支持采样长度为 L+M 的连续序列（用于 Burn-in 训练）
    - 不存储 hidden state（每个 batch 独立初始化）
    """

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1000000):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """存储单个 transition"""
        # 形状检查：防止广播错误 (72,) -> (12,)
        assert np.isscalar(reward) or reward.shape == () or reward.shape == (1,), f"Reward shape error: {np.shape(reward)}"
        assert state.shape == (self.state_buf.shape[1],), f"State shape error: {state.shape}, expected {(self.state_buf.shape[1],)}"
        assert next_state.shape == (self.next_state_buf.shape[1],), f"Next state shape error: {next_state.shape}, expected {(self.next_state_buf.shape[1],)}"
        assert action.shape == (self.action_buf.shape[1],), f"Action shape error: {action.shape}, expected {(self.action_buf.shape[1],)}"

        idx = self.ptr % self.capacity
        self.state_buf[idx] = state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = next_state
        self.done_buf[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample_sequence(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        采样长度为 seq_len 的连续序列
        
        返回:
            - states: [batch_size, seq_len, state_dim]
            - actions: [batch_size, seq_len, action_dim]
            - rewards: [batch_size, seq_len, 1]
            - next_states: [batch_size, seq_len, state_dim]
            - dones: [batch_size, seq_len, 1]
        """
        # 确保有足够的样本
        if self.size < seq_len:
            raise ValueError(f"Buffer size {self.size} < sequence length {seq_len}")
        
        # 随机选择起始位置
        valid_start = self.size - seq_len
        
        batch_indices = []
        while len(batch_indices) < batch_size:
            start_idx = np.random.randint(0, max(1, valid_start))
            # 检查序列中是否包含 episode 结束（除了最后一步）
            if np.any(self.done_buf[start_idx : start_idx + seq_len - 1]):
                continue  # 如果序列跨越了 episode 边界，则重新采样
            batch_indices.append(start_idx)
        start_idxs = np.array(batch_indices)

        # 提取连续序列
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for start_idx in start_idxs:
            end_idx = start_idx + seq_len
            seq_states = self.state_buf[start_idx:end_idx]
            seq_actions = self.action_buf[start_idx:end_idx]
            seq_rewards = self.reward_buf[start_idx:end_idx]
            seq_next_states = self.next_state_buf[start_idx:end_idx]
            seq_dones = self.done_buf[start_idx:end_idx]
            
            states.append(seq_states)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
        return {
            "states": torch.as_tensor(np.array(states), device=device, dtype=torch.float32),
            "actions": torch.as_tensor(np.array(actions), device=device, dtype=torch.float32),
            "rewards": torch.as_tensor(np.array(rewards), device=device, dtype=torch.float32),
            "next_states": torch.as_tensor(np.array(next_states), device=device, dtype=torch.float32),
            "dones": torch.as_tensor(np.array(dones), device=device, dtype=torch.float32),
        }


# =============================================================
# 物理一致性无人机环境包装器 (Physics-Consistent Wrapper)
# =============================================================

class PhysicsDroneWrapper(gym.Wrapper):
    """
    将 gym-pybullet-drones 包装为符合 TD3-Latent 要求的物理接口
    核心功能：
    1. 动作映射：Actor 输出期望加速度 -> 语义翻译层 -> 电机 RPM
    2. 状态处理：确保输出 12 维 Euler 状态
    3. 奖励计算：位置、姿态、平滑度
    """
    def __init__(self, env):
        super().__init__(env)
        self.mass = 0.027  # CrazyFlie 质量 (kg)
        self.g = 9.81
        self.kf = 3.16e-10 # 电机推力系数
        self.km = 7.94e-12 # 电机力矩系数
        self.arm_length = 0.0397 # 机臂长度
        self.max_rpm = 22000 # 最大转速
        self.last_action = np.zeros(4)
        self.last_obs = None
        self.debug_info = {}

        # 状态归一化参数 (静态)
        self.obs_mean = np.array([
            0., 0., 1.,  # pos ~ [0,0,1]
            0., 0., 0.,  # rpy
            0., 0., 0.,  # vel
            0., 0., 0.   # ang_vel
        ], dtype=np.float32)
        self.obs_std = np.array([
            0.5, 0.5, 0.5, # pos
            np.pi, np.pi, np.pi, # rpy
            2.0, 2.0, 2.0, # vel
            4.0, 4.0, 4.0  # ang_vel
        ], dtype=np.float32)
        # 防止除以零
        self.obs_std[self.obs_std == 0] = 1.0
        
        # 动作空间：[ax, ay, az, yaw_rate] (期望加速度 + 偏航角速度)
        # 范围由 Actor 的 action_max 控制 (例如 +/- 10 m/s^2)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # 强制状态空间为 12 维（只包含运动学部分），避免 ReplayBuffer 和网络维度不一致
        # 这也确保 env.observation_space 与 _process_obs 返回的一致。
        self.state_dim = 12
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_action = np.zeros(4)
        processed_obs = self._process_obs(obs)
        # 🔴 断言1：环境输出应该始终是 12 维
        assert processed_obs.shape == (self.state_dim,), f"Env.reset returned wrong obs {processed_obs.shape}"
        return processed_obs, info

    def step(self, action):
        """
        输入 action: [ax, ay, az, yaw_rate] (期望加速度 + 偏航速率)
        语义：
            ax, ay, az: 世界坐标系下的期望加速度 (m/s^2)
            yaw_rate: 机体坐标系下的期望偏航角速度 (rad/s)
        """
        # 确保 action 是 1D 数组 (4,)，防止广播错误
        action = np.asarray(action).reshape(-1)
        assert action.shape == (4,), f"Action shape mismatch: {action.shape} != (4,)"

        # 1. 语义翻译层 (Semantic Translation Layer)
        # 将高层物理量 (a_des) 转换为底层控制量 (RPM)
        rpms = self.accel_to_rpm(action, self.last_obs)
        
        # 2. 环境步进
        obs, _, terminated, truncated, info = self.env.step(rpms)

        # 存储调试信息
        self.debug_info['raw_env_obs'] = obs.copy()
        self.debug_info['a_des_input'] = action.copy()
        self.debug_info['final_rpms'] = rpms.copy()

        self.last_obs = obs
        processed_obs = self._process_obs(obs)

        # 🔴 断言1：环境输出应该始终是 12 维
        assert processed_obs.shape == (self.state_dim,), f"Env.step returned wrong obs {processed_obs.shape}"

        self.debug_info['processed_obs_for_rl'] = processed_obs.copy()
        
        # 3. 奖励计算 (Reward Engineering)
        reward = self._compute_reward(obs, action) # 使用原始 obs 计算奖励
        
        self.last_action = action.copy()
        return processed_obs, reward, terminated, truncated, info

    def accel_to_rpm(self, action, obs):
        """
        将期望加速度转换为电机 RPM
        action: [ax, ay, az, yaw_rate]
        """
        # 提取当前姿态 (假设 obs 是 KIN 模式: [x,y,z, r,p,y, ...])
        # 注意：这里需要根据实际 obs 结构提取 r, p, y
        # HoverAviary KIN: 0-2 pos, 3-5 rpy, 6-8 vel, 9-11 ang_vel
        r, p, y = obs.reshape(-1)[3:6]

        # 1. 计算期望推力向量 (世界坐标系)
        # F_des = m * (a_des + g)
        # action[:3] 是期望加速度，我们需要加上重力补偿
        # 注意：Actor 输出的 action[2] 通常已经是包含或不包含重力的，这里假设 Actor 输出的是纯加速度 a_des
        # 我们需要加上重力向量 [0, 0, 9.81] 才能维持悬停
        a_des = action[:3]
        total_acc_des = a_des + np.array([0, 0, self.g])
        f_des_world = self.mass * total_acc_des

        # 2. 将推力向量投影到机体坐标系
        # 构建旋转矩阵 R (Body -> World)
        # 简化计算：z_body 在世界坐标系下的向量
        # z_b = [cos(y)sin(p)cos(r) + sin(y)sin(r), sin(y)sin(p)cos(r) - cos(y)sin(r), cos(p)cos(r)]
        # 这里使用近似或简单的旋转矩阵转置 R^T * F_world
        
        # 简单起见，我们计算需要的推力大小和力矩
        # 推力大小 T = F_des_world dot z_body_current
        # 这是一个简化的映射，假设无人机姿态控制由力矩完成
        
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)
        
        # Z-Y-X 旋转矩阵的第三列 (z_body)
        z_body = np.array([
            cy*sp*cr + sy*sr,
            sy*sp*cr - cy*sr,
            cp*cr
        ])
        
        # 期望推力 (标量)
        thrust_mag = np.dot(f_des_world, z_body)
        
        # 3. 计算期望力矩 (用于姿态调整)
        # 我们希望 z_body 对齐 f_des_world 的方向
        # 使用叉乘计算旋转误差: error = z_body x (f_des / |f_des|)
        f_dir = f_des_world / (np.linalg.norm(f_des_world) + 1e-6)
        rot_err = np.cross(z_body, f_dir)
        
        # 简单的 P 控制器将旋转误差映射为力矩
        # 增益需要根据机体惯量调整，这里给出一个经验值
        k_att = 0.02 
        torque_x = k_att * rot_err[0] # Roll 力矩 (近似，基于世界系误差投影)
        torque_y = k_att * rot_err[1] # Pitch 力矩
        
        # 实际上 rot_err 是世界系的，应该转到机体系，但对于微小误差，直接映射也行
        # 更严谨的做法是 R.T @ rot_err
        
        # 偏航力矩直接使用 action[3] (yaw_rate)
        # 假设 action[3] 是期望的偏航角速度，这里简化为直接作为力矩的前馈
        torque_z = 0.001 * action[3]

        # 4. 混控 (Mixing) -> RPM
        # CrazyFlie X 模式混控
        # PWM_0 = T - Tx - Ty - Tz
        # PWM_1 = T - Tx + Ty + Tz
        # PWM_2 = T + Tx + Ty - Tz
        # PWM_3 = T + Tx - Ty + Tz
        # 转换为 RPM^2 关系
        
        t = thrust_mag / (4 * self.kf)
        r = torque_x / (4 * self.kf * self.arm_length) # Roll 力臂
        p = torque_y / (4 * self.kf * self.arm_length) # Pitch 力臂
        y = torque_z / (4 * self.km)

        rpm_sq = np.array([
            t - r - p - y,
            t - r + p + y,
            t + r + p - y,
            t + r - p + y
        ])
        
        rpms = np.sqrt(np.maximum(0, rpm_sq))
        rpms = np.clip(rpms, 0, self.max_rpm)

        # 适配 gym-pybullet-drones 的动作维度要求 (NUM_DRONES, 4)
        rpms = rpms.reshape(1, -1)
        return rpms

    def _process_obs(self, obs):
        """
        观测预处理：
        - 只对前 12 维 KIN 状态做归一化
        - 后面的额外维度（例如历史动作、辅助特征）保持原样，避免维度不匹配
        """
        obs = obs.reshape(-1)

        # 前 12 维：HoverAviary KIN 默认: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]
        kin = obs[:12]
        kin_norm = (kin - self.obs_mean) / self.obs_std

        # 如果存在额外维度（例如 72 维状态的后 60 维），直接拼接回去
        # if obs.shape[0] > 12:
        #     extra = obs[12:]
        #     normalized_obs = np.concatenate([kin_norm, extra], axis=0)
        # else:
        #     normalized_obs = kin_norm

        # 直接返回12维归一化状态
        normalized_obs = kin_norm
    
         # 🔴 添加断言确保是12维
        assert normalized_obs.shape[0] == 12, f"_process_obs 返回了 {normalized_obs.shape} 维，应该是12维"

        return normalized_obs

    def _compute_reward(self, obs, action):
        # obs 是原始 (未归一化) 的观测值
        # obs is raw (unnormalized) observation
        pos = obs.reshape(-1)[0:3]
        rpy = obs.reshape(-1)[3:6]
        target = np.array([0, 0, 1.0]) # 悬停目标
        
        r_pos = -1.0 * np.linalg.norm(pos - target)**2
        r_att = -0.1 * np.linalg.norm(rpy)**2
        r_jerk = -0.05 * np.linalg.norm(action - self.last_action)**2
        r_survive = 0.1 if pos[2] > 0.1 and np.abs(rpy[0]) < 1.0 and np.abs(rpy[1]) < 1.0 else 0.0
        
        return r_pos + r_att + r_jerk + r_survive


# =============================================================
# TD3 超参数配置（UAV 版本）
# =============================================================

@dataclass
class TD3UAVConfig:
    state_dim: int
    action_dim: int
    action_max: float
    
    # 状态拆分参数
    high_state_dim: int = 0  # 60变为0
    low_state_dim: int = 12   # 当前运动学状态维度
    enable_state_split: bool = False  # 不启用状态拆分
    
    # 网络结构参数
    latent_dim: int = 32           # 潜变量维度（增大以防止内环欠拟合）
    gru_hidden_dim: int = 256      # GRU 隐藏层维度
    hidden_dim: int = 512          # Actor/Critic 网络隐藏层维度
    
    # 频率适配参数
    encoder_interval: int = 8      # Encoder 更新间隔（1/8 Decoder 频率）
    
    # 序列训练参数（Burn-in）
    burn_in_length: int = 8        # Burn-in 序列长度（L）
    learning_length: int = 16      # Learning 序列长度（M）
    enable_burn_in: bool = True    # 是否启用 Burn-in 训练
    
    # TD3 基础参数
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2          # TD3 的 delayed policy update
    # Dynamics / regularization 权重
    lambda_hover: float = 0.1     # 悬停正则化权重（用于 Actor loss）


# =============================================================
# TD3-Latent UAV 主体
# =============================================================

class TD3LatentUAV:
    """
    TD3 + 层级异步控制（UAV 版本）
    
    核心特性：
    1. 状态语义拆分：s_high（任务级）+ s_low（执行级）
    2. Burn-in 序列训练：让 GRU 记忆真正生效
    3. 动作低通滤波：执行器物理约束
    4. 频率适配：Encoder 1/8 Decoder 频率的语义异步控制
    """

    def __init__(self, config: TD3UAVConfig, device: torch.device):
        self.device = device
        self.config = config
        
        # 强制使用 Config 中的维度设置 (UAV 专用)
        high_dim = config.high_state_dim
        low_dim = config.low_state_dim
        self.high_state_dim = high_dim
        self.low_state_dim = low_dim

        # -------- Actor（高层 GRU Encoder + 低层 Decoder）--------
        self.actor_encoder = ActorEncoder(
            high_dim,
            config.latent_dim,
            config.gru_hidden_dim
        ).to(device)
        self.actor_encoder_target = ActorEncoder(
            high_dim,
            config.latent_dim,
            config.gru_hidden_dim
        ).to(device)

        self.actor_decoder = ActorDecoder(
            low_dim,
            config.action_dim,
            config.latent_dim,
            config.action_max,
            config.hidden_dim,
        ).to(device)
        self.actor_decoder_target = ActorDecoder(
            low_dim,
            config.action_dim,
            config.latent_dim,
            config.action_max,
            config.hidden_dim,
        ).to(device)

        # -------- 双 Q 网络（TD3 核心）--------
        self.critic1 = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic2 = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic1_target = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic2_target = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)

        # -------- Optimizer（分离优化器）--------
        self.encoder_opt = torch.optim.Adam(self.actor_encoder.parameters(), lr=config.actor_lr)
        self.decoder_opt = torch.optim.Adam(self.actor_decoder.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_lr,
        )

        # -------- Dynamics Model（最小前向动力学）--------
        # Dynamics 输入为 (s_low + action)，预测 next_s_low
        self.dynamics = DynamicsModel(
            low_state_dim=self.low_state_dim,
            action_dim=config.action_dim,
            pred_low_dim=self.low_state_dim,
            hidden_dim=config.hidden_dim
        ).to(device)
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=config.critic_lr)

        # hover 权重
        self.lambda_hover = config.lambda_hover

        # 物理常量（用于 hover 正则化计算）
        self.mass = 0.027
        self.g = 9.81

        # -------- 异步更新相关状态（用于交互）--------
        self.current_latent = None
        self.encoder_hidden = None
        self.debug_info = {}
        self.step_count = 0

        # 初始化 target 网络
        self._hard_update_all()
        self.total_it = 0

    def _hard_update_all(self):
        """硬更新所有 target 网络"""
        for target, src in [
            (self.actor_encoder_target, self.actor_encoder),
            (self.actor_decoder_target, self.actor_decoder),
            (self.critic1_target, self.critic1),
            (self.critic2_target, self.critic2),
        ]:
            target.load_state_dict(src.state_dict())

    def split_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """拆分状态为 s_high 和 s_low"""
        if not self.config.enable_state_split:
            # 如果不启用状态拆分，返回完整状态作为 s_low，空数组作为 s_high
            return np.array([]), state
        
        return split_state_uav(state, self.config.state_dim)

    def select_action(self, state: np.ndarray, noise_std: float = 0.0, reset_hidden: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # 🔴 断言3：检查Actor输入
        assert state.shape[0] == 12, f"select_action 输入state维度错误: {state.shape}"
        """
        选择动作，实现语义异步控制
        
        返回:
            action: 期望加速度/推力向量 (a_des)
        """
        # 拆分状态
        s_high, s_low = self.split_state(state)
        s_high_t = torch.as_tensor(s_high, device=self.device, dtype=torch.float32).unsqueeze(0) if len(s_high) > 0 else None
        s_low_t = torch.as_tensor(s_low, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # 如果重置 hidden 或首次调用，初始化 hidden state
        if reset_hidden or self.encoder_hidden is None:
            if s_high_t is not None:
                self.encoder_hidden = self.actor_encoder.init_hidden(1, device=self.device)
            self.step_count = 0
        
        # 异步更新：每隔 encoder_interval 步更新一次 latent
        if self.step_count % self.config.encoder_interval == 0 or self.current_latent is None:
            with torch.no_grad():
                if s_high_t is not None:
                    self.debug_info['encoder_input_shigh'] = s_high_t.cpu().numpy()
                    latent, self.encoder_hidden = self.actor_encoder(s_high_t, self.encoder_hidden)
                    self.current_latent = latent
                    self.debug_info['encoder_output_latent'] = latent.cpu().numpy()
                    # latent norm 诊断
                    try:
                        self.debug_info['latent_norm'] = float(torch.norm(self.current_latent).cpu().item())
                    except Exception:
                        self.debug_info['latent_norm'] = None
                else:
                    # 如果未启用状态拆分，使用零 latent
                    self.current_latent = torch.zeros(1, self.config.latent_dim, device=self.device)
                    self.debug_info['encoder_input_shigh'] = "None (split disabled)"
                    self.debug_info['encoder_output_latent'] = self.current_latent.cpu().numpy()
        
        self.step_count += 1
        
        # 记录 Decoder 输入
        self.debug_info['decoder_input_latent'] = self.current_latent.cpu().numpy()
        self.debug_info['decoder_input_slow'] = s_low_t.cpu().numpy()
        
        # 使用当前 latent 解码动作（内环执行器）
        with torch.no_grad():
            raw_action = self.actor_decoder(self.current_latent, s_low_t)

        # 记录 Decoder 输出
        self.debug_info['decoder_output_raw_action'] = raw_action.cpu().numpy()

        # latent norm (redundant safety)
        try:
            self.debug_info['latent_norm'] = float(torch.norm(self.current_latent).cpu().item())
        except Exception:
            self.debug_info['latent_norm'] = None

        action = raw_action.cpu().numpy()[0]

        # 探索噪声（添加到原始动作）
        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action, -self.config.action_max, self.config.action_max)

        return action

    def train_step(self, replay_buffer: SequenceReplayBuffer, batch_size: int):
        """训练一步，支持 Burn-in 序列训练"""
        if self.config.enable_burn_in:
            # Burn-in 序列训练模式
            seq_len = self.config.burn_in_length + self.config.learning_length
            if replay_buffer.size < seq_len:
                return {}
            
            batch = replay_buffer.sample_sequence(batch_size, seq_len, self.device)
            
            states = batch["states"]  # [batch_size, seq_len, state_dim]
            actions = batch["actions"]  # [batch_size, seq_len, action_dim]
            rewards = batch["rewards"]  # [batch_size, seq_len, 1]
            next_states = batch["next_states"]  # [batch_size, seq_len, state_dim]
            dones = batch["dones"]  # [batch_size, seq_len, 1]

            action_mean = actions.mean().item()
            action_std = actions.std().item()
            print(f"Train step - Action mean: {action_mean:.4f}, std: {action_std:.4f}")
            
            # 拆分状态序列（批量版本，避免维度错误）
            # states/next_states: [batch_size, seq_len, state_dim]
            s_high_seq: List[Optional[torch.Tensor]] = []
            s_low_seq: List[torch.Tensor] = []
            next_s_high_seq: List[Optional[torch.Tensor]] = []
            next_s_low_seq: List[torch.Tensor] = []
            
            for i in range(seq_len):
                s = states[:, i, :]       # [batch_size, state_dim]
                ns = next_states[:, i, :] # [batch_size, state_dim]

                if self.config.enable_state_split:
                    # 新架构拆分逻辑: s_low 是运动学, s_high 是历史
                    s_low = s[:, :12]
                    s_high = s[:, 12:]
                    
                    ns_low = ns[:, :12]
                    ns_high = ns[:, 12:]

                    s_high_seq.append(s_high)
                    s_low_seq.append(s_low)
                    next_s_high_seq.append(ns_high)
                    next_s_low_seq.append(ns_low)
                else:
                    # 不启用状态拆分：s_high 为空，s_low 直接使用完整状态
                    s_high_seq.append(None)
                    s_low_seq.append(s)
                    next_s_high_seq.append(None)
                    next_s_low_seq.append(ns)
            
            # Burn-in 阶段：预热 GRU hidden state（不计算 loss）
            hidden = None
            for i in range(self.config.burn_in_length):
                if s_high_seq[i] is not None:
                    _, hidden = self.actor_encoder(s_high_seq[i], hidden)
                else:
                    hidden = self.actor_encoder.init_hidden(batch_size, device=self.device)
            
            # Learning 阶段：使用 Burn-in 后的 hidden state 进行训练
            # 使用最后一步作为训练样本（可以扩展到对所有 learning 步骤训练）
            learning_start = self.config.burn_in_length
            train_idx = learning_start  # 使用 learning 阶段的第一步作为训练样本
            
            # 为 Critic 准备 target（使用 target network）
            with torch.no_grad():
                target_hidden = hidden.clone() if hidden is not None else None
                # Burn-in target network
                for j in range(learning_start):
                    if next_s_high_seq[j] is not None:
                        _, target_hidden = self.actor_encoder_target(next_s_high_seq[j], target_hidden)
                    else:
                        target_hidden = self.actor_encoder_target.init_hidden(batch_size, device=self.device)
                
                # 计算 target action 和 target Q
                if next_s_high_seq[train_idx] is not None:
                    next_latent, _ = self.actor_encoder_target(next_s_high_seq[train_idx], target_hidden)
                else:
                    next_latent = torch.zeros(batch_size, self.config.latent_dim, device=self.device)
                next_action = self.actor_decoder_target(next_latent, next_s_low_seq[train_idx])
                
                # TD3 target policy smoothing
                noise = (torch.randn_like(next_action) * self.config.policy_noise).clamp(
                    -self.config.noise_clip, self.config.noise_clip
                )
                next_action = (next_action + noise).clamp(
                    -self.config.action_max, self.config.action_max
                )
                
                target_q = torch.min(
                    self.critic1_target(next_states[:, train_idx], next_action),
                    self.critic2_target(next_states[:, train_idx], next_action),
                )
                target = rewards[:, train_idx] + (1 - dones[:, train_idx]) * self.config.gamma * target_q
            
            # Critic 更新（使用完整状态和原始动作）
            current_q1 = self.critic1(states[:, train_idx], actions[:, train_idx])
            current_q2 = self.critic2(states[:, train_idx], actions[:, train_idx])
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
            # Q 值的统计信息
            try:
                q_vals = torch.cat([current_q1.view(-1), current_q2.view(-1)], dim=0)
                info_q_mean = float(q_vals.mean().item())
                info_q_std = float(q_vals.std().item())
                self.debug_info['q_mean'] = info_q_mean
                self.debug_info['q_std'] = info_q_std
            except Exception:
                self.debug_info['q_mean'] = None
                self.debug_info['q_std'] = None
            
            # 记录 Critic 调试信息 (batch 中第一个样本)
            self.debug_info['critic_input_state'] = states[:, train_idx][0].cpu().numpy()
            self.debug_info['critic_input_action'] = actions[:, train_idx][0].cpu().numpy()
            self.debug_info['critic_output_q1'] = current_q1[0].item()
            
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
            )
            self.critic_opt.step()
            
            info = {"critic_loss": critic_loss.item()}

            # ------------------ Dynamics loss & update ------------------
            try:
                dyn_input = torch.cat([s_low_seq[train_idx].detach(), actions[:, train_idx].detach()], dim=-1)
                pred_next_low = self.dynamics(dyn_input)
                true_next_low = next_s_low_seq[train_idx]
                dyn_loss = F.mse_loss(pred_next_low, true_next_low)

                self.dynamics_opt.zero_grad()
                dyn_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
                self.dynamics_opt.step()

                info["dyn_loss"] = dyn_loss.item()
                # 记录到调试信息
                self.debug_info['dyn_pred_error'] = float(dyn_loss.item())
            except Exception as e:
                # 保持鲁棒性，避免训练中断
                info["dyn_loss"] = None
                self.debug_info['dyn_pred_error'] = str(e)
            
            # Actor 更新（Delayed Policy Update）
            if self.total_it % self.config.policy_delay == 0:
                # 重新通过整个序列计算 latent（保持梯度连接）
                # 重要：整个序列都需要参与梯度计算，以确保 GRU 能够学习时序依赖
                train_hidden = None
                
                # Burn-in 阶段：前向传播（需要梯度，但不计算 loss）
                # 这样 GRU 参数可以通过整个序列反向传播
                burn_in_latents = []
                for j in range(learning_start):
                    if s_high_seq[j] is not None:
                        latent_j, train_hidden = self.actor_encoder(s_high_seq[j], train_hidden)
                        burn_in_latents.append(latent_j)
                    else:
                        train_hidden = self.actor_encoder.init_hidden(batch_size, device=self.device)
                        burn_in_latents.append(None)
                
                # Learning 阶段：使用 Burn-in 后的 hidden state（需要梯度）
                if s_high_seq[train_idx] is not None:
                    latent, _ = self.actor_encoder(s_high_seq[train_idx], train_hidden)
                else:
                    latent = torch.zeros(batch_size, self.config.latent_dim, device=self.device, requires_grad=True)
                
                current_action = self.actor_decoder(latent, s_low_seq[train_idx])
                
                # TD3 Actor loss（最大化 Q 值）
                actor_loss = -self.critic1(states[:, train_idx], current_action).mean()

                # latent norm 诊断（batch 平均）
                try:
                    latent_norm = float(torch.norm(latent, dim=1).mean().cpu().item())
                    info['latent_norm'] = latent_norm
                    self.debug_info['latent_norm'] = latent_norm
                except Exception:
                    info['latent_norm'] = None

                # ------------------ Hover 正则化（鼓励总推力接近 mg） ------------------
                try:
                    # s_low_seq[train_idx] 包含 r,p,y 在索引 3:6
                    s_low = s_low_seq[train_idx]
                    rpy = s_low[:, 3:6]
                    roll = rpy[:, 0]
                    pitch = rpy[:, 1]
                    yaw = rpy[:, 2]

                    cy = torch.cos(yaw)
                    sy = torch.sin(yaw)
                    cp = torch.cos(pitch)
                    sp = torch.sin(pitch)
                    cr = torch.cos(roll)
                    sr = torch.sin(roll)

                    # z_body (batch, 3)
                    z_x = cy * sp * cr + sy * sr
                    z_y = sy * sp * cr - cy * sr
                    z_z = cp * cr
                    z_body = torch.stack([z_x, z_y, z_z], dim=-1)

                    # total desired acceleration (包含重力补偿)
                    total_acc_des = current_action[:, :3] + torch.tensor([0.0, 0.0, self.g], device=self.device)
                    f_des_world = self.mass * total_acc_des

                    thrust_mag = (f_des_world * z_body).sum(dim=-1)  # (batch,)
                    # 目标 thrust 为 mg
                    mg = self.mass * self.g
                    hover_loss = F.mse_loss(thrust_mag, torch.full_like(thrust_mag, mg))

                    # 将 hover 正则加入 actor_loss（作为惩罚项）
                    actor_loss = actor_loss + self.lambda_hover * hover_loss

                    info['hover_loss'] = float(hover_loss.item())
                    self.debug_info['hover_loss'] = float(hover_loss.item())
                    self.debug_info['avg_thrust'] = float(thrust_mag.mean().item())
                except Exception as e:
                    info['hover_loss'] = None
                    self.debug_info['hover_loss'] = str(e)

                # 记录动作饱和（估计）：检查原始动作是否接近动作上限
                try:
                    sat_pct = (torch.abs(current_action) >= (self.config.action_max * 0.99)).float().mean()
                    info['saturation_pct'] = float(sat_pct.item())
                    self.debug_info['saturation_pct'] = float(sat_pct.item())
                except Exception:
                    info['saturation_pct'] = None
                    self.debug_info['saturation_pct'] = None
                
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_encoder.parameters()) + list(self.actor_decoder.parameters()), 1.0
                )
                self.encoder_opt.step()
                self.decoder_opt.step()
                
                # 软更新 target 网络
                self._soft_update(self.actor_encoder_target, self.actor_encoder)
                self._soft_update(self.actor_decoder_target, self.actor_decoder)
                self._soft_update(self.critic1_target, self.critic1)
                self._soft_update(self.critic2_target, self.critic2)
                
                info["actor_loss"] = actor_loss.item()
            
            self.total_it += 1
            return info
            
        else:
            # fallback: 普通 TD3
            batch = replay_buffer.sample(batch_size)

            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]

    def _soft_update(self, target: nn.Module, src: nn.Module):
        """软更新（Polyak Averaging）"""
        tau = self.config.tau
        for t_param, param in zip(target.parameters(), src.parameters()):
            t_param.data.mul_(1 - tau).add_(tau * param.data)


# =============================================================
# 评估函数
# =============================================================

def evaluate(env: gym.Env, agent: TD3LatentUAV, episodes: int = 10, noise: float = 0.0):
    """评估智能体性能"""
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_ret = 0.0
        episode_step = 0
        
        while not done:
            reset_hidden = (episode_step == 0)
            action = agent.select_action(state, noise_std=noise, reset_hidden=reset_hidden)
            state, reward, terminated, truncated, _ = env.step(action)
            # 🔴 断言1：环境输出应该保持 12 维
            assert state.shape == (12,), f"Env returned wrong obs {state.shape}"
            done = terminated or truncated
            ep_ret += reward
            episode_step += 1
        
        returns.append(ep_ret)
    return float(np.mean(returns))


# =============================================================
# 训练主函数
# =============================================================

def train(args):
    # 环境替换为 HoverAviary 并应用物理包装器
    # HoverAviary 默认是单机环境，不支持 num_drones 参数
    env = PhysicsDroneWrapper(HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM,
        gui=False
    ))
    eval_env = PhysicsDroneWrapper(HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM,
        gui=False
    ))

    writer = SummaryWriter(log_dir=f"runs/TD3_Latent_UAV_{args.seed}")

    state_dim = env.observation_space.shape[0]
    assert state_dim == 12, f"Expected state_dim==12 but got {state_dim}. Ensure env observation space matches processed obs."
    print(f"Corrected state_dim: {state_dim}") # 通常应该是 12
    action_dim = env.action_space.shape[0]
    # action_max = float(env.action_space.high[0]) # 原始是 1.0
    action_max = 10.0 # 调整为加速度的物理范围 (m/s^2)

    # 警告：根据用户请求，我们假设环境已被修改以输出72维状态。
    # 如果环境仍然是12维，这里会出错。

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    config = TD3UAVConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_max=action_max,
        high_state_dim=0,# 改为0，因为不再有历史序列
        low_state_dim=state_dim, # 直接使用完整状态作为 s_low
        enable_state_split=False, # 关闭状态拆分，直接使用完整状态
        latent_dim=args.latent_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        hidden_dim=args.hidden_dim,
        encoder_interval=args.encoder_interval,
        burn_in_length=args.burn_in_length,
        learning_length=args.learning_length,
        enable_burn_in=args.enable_burn_in,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        policy_delay=args.policy_delay,
    )
    
    agent = TD3LatentUAV(config, device)
    buffer = SequenceReplayBuffer(state_dim, action_dim, capacity=args.buffer_size)

    state, _ = env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    episode_reward = 0.0
    episode_step = 0
    log_rewards = []

    try:
        for t in range(1, args.total_steps + 1):
            # 计算噪声衰减
            expl_noise = max(
                args.expl_noise_end,
                args.expl_noise_start - (t / args.noise_decay_steps) * (args.expl_noise_start - args.expl_noise_end)
            )

            if t < args.start_steps:
                # 随机采样时也需要符合加速度范围
                action = np.random.uniform(-action_max, action_max, size=(action_dim,))
            else:
                reset_hidden = (episode_step == 0)
                action = agent.select_action(state, noise_std=expl_noise, reset_hidden=reset_hidden)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储原始动作到 ReplayBuffer（Critic 评估用）
            buffer.push(state, action, reward, next_state, float(done))

            if t % 1000 == 0:  # 每1000步打印一次
                print(f"Step {t}: ReplayBuffer size = {buffer.size}")

            state = next_state
            episode_reward += reward
            episode_step += 1

            # 训练
            if t >= args.update_after:
                print(f"Step {t}: Calling train_step (buffer size {buffer.size})")
                info = agent.train_step(buffer, args.batch_size)
                if info:  # 训练实际执行了
                    print(f"  -> Training executed, losses: { {k: v for k, v in info.items() if v is not None} }")      
                info = agent.train_step(buffer, args.batch_size)

                # 🔴 修改为安全打印
                if t % 100 == 0:
                    if info:  # info不为空时才记录
                        for key, value in info.items():
                            if value is not None:  # 值为None时不记录
                                writer.add_scalar(f"Loss/{key}", value, t)
                    else:
                        print(f"Step {t}: 训练尚未开始（buffer size: {buffer.size}）")
                    writer.add_scalar("Stats/Exploration_Noise", expl_noise, t)
           
        
            # 处理 Episode 结束
            if done:
                writer.add_scalar("Reward/Train_Episode_Reward", episode_reward, t)
                log_rewards.append(episode_reward)
                
                state, _ = env.reset()
                episode_reward = 0.0
                episode_step = 0

            # 评估
            if t % args.eval_interval == 0:
                avg_eval_ret = evaluate(eval_env, agent, episodes=args.eval_episodes)
                writer.add_scalar("Reward/Eval_Average_Return", avg_eval_ret, t)
                
                # =============================================================
                # DETAILED DEBUG LOGGING
                # =============================================================
                print("\n" + "="*20 + f" DEBUG SUMMARY @ STEP {t} " + "="*20)

                # concise env / RL summary
                pos = env.debug_info.get('processed_obs_for_rl', None)
                pos_str = f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})" if isinstance(pos, np.ndarray) and pos.size>=3 else "pos=N/A"

                # Q stats
                q_mean = agent.debug_info.get('q_mean', None)
                q_std = agent.debug_info.get('q_std', None)

                # dynamics / hover / saturation
                dyn_err = agent.debug_info.get('dyn_pred_error', None)
                hover_l = agent.debug_info.get('hover_loss', None)
                avg_thrust = agent.debug_info.get('avg_thrust', None)
                sat_pct = agent.debug_info.get('saturation_pct', None)
                latent_norm = agent.debug_info.get('latent_norm', None)

                # Safe formatting: some debug fields may be None (e.g., before first update)
                avg_eval_str = f"{avg_eval_ret:.2f}" if isinstance(avg_eval_ret, (int, float)) else str(avg_eval_ret)
                q_mean_str = f"{q_mean:.3f}" if isinstance(q_mean, (int, float)) else str(q_mean)
                q_std_str = f"{q_std:.3f}" if isinstance(q_std, (int, float)) else str(q_std)
                dyn_err_str = f"{dyn_err:.4f}" if isinstance(dyn_err, (int, float)) else str(dyn_err)
                hover_l_str = f"{hover_l:.4f}" if isinstance(hover_l, (int, float)) else str(hover_l)

                latent_norm_str = f"{latent_norm:.3f}" if isinstance(latent_norm, (int, float)) else str(latent_norm)
                avg_thrust_str = f"{avg_thrust:.3f}" if isinstance(avg_thrust, (int, float)) else str(avg_thrust)
                sat_pct_str = f"{sat_pct:.3f}" if isinstance(sat_pct, (int, float)) else str(sat_pct)

                # 🔴 修改为安全打印
                print(f"Step {t} | EvalRet {avg_eval_ret if avg_eval_ret is not None else 'N/A':.2f} | ", end="")
                print(f"Q mean: {q_mean if q_mean is not None else 'N/A'} | ", end="")
                print(f"dyn_loss: {dyn_err if dyn_err is not None else 'N/A'}")
                
                print(f"{pos_str} | latent_norm: {latent_norm_str} | avg_thrust: {avg_thrust_str} | action_sat_pct: {sat_pct_str}")

                # motor RPM saturation (if available)
                if 'final_rpms' in env.debug_info:
                    try:
                        rpms = np.asarray(env.debug_info['final_rpms']).flatten()
                        maxrpm = getattr(env, 'max_rpm', None) or 1.0
                        sat_per_motor = (rpms / maxrpm).clip(0,1)
                        print(f"Motor RPMs: {np.round(rpms,0).tolist()} | sat% per motor: {np.round(sat_per_motor*100,1).tolist()}%")
                    except Exception:
                        pass

                print("="*66 + "\n")
                
    except KeyboardInterrupt:
        print("\n检测到用户中断...")

    writer.close()
    return agent, log_rewards


# =============================================================
# 命令行参数解析
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TD3-Latent UAV (Hierarchical Control) for HalfCheetah-v5")
    
    # 状态拆分参数
    parser.add_argument("--high_state_dim", type=int, default=60, help="s_high 维度 (历史序列)")
    parser.add_argument("--low_state_dim", type=int, default=12, help="s_low 维度 (当前运动学)")
    parser.add_argument("--enable_state_split", action="store_true", default=True, help="启用状态语义拆分（默认启用）")
    parser.add_argument("--disable_state_split", dest="enable_state_split", action="store_false", help="禁用状态语义拆分")
    
    # 网络结构参数
    parser.add_argument("--latent_dim", type=int, default=32, help="潜变量维度（≥32 以防止内环欠拟合）")
    parser.add_argument("--gru_hidden_dim", type=int, default=256, help="GRU 隐藏层维度")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Actor/Critic 网络隐藏层维度")
    
    # 频率适配参数
    parser.add_argument("--encoder_interval", type=int, default=8, help="Encoder 更新间隔（1/8 Decoder 频率）")
    
    # 序列训练参数（Burn-in）
    parser.add_argument("--burn_in_length", type=int, default=8, help="Burn-in 序列长度（L）")
    parser.add_argument("--learning_length", type=int, default=16, help="Learning 序列长度（M）")
    parser.add_argument("--enable_burn_in", action="store_true", default=True, help="启用 Burn-in 序列训练（默认启用）")
    parser.add_argument("--disable_burn_in", dest="enable_burn_in", action="store_false", help="禁用 Burn-in 序列训练")
    
    # TD3 基础参数
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor 学习率")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic 学习率")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    parser.add_argument("--policy_delay", type=int, default=2, help="TD3 延迟策略更新间隔")
    
    # 训练参数
    parser.add_argument("--total_steps", type=int, default=200_000, help="总训练步数")
    parser.add_argument("--start_steps", type=int, default=10_000, help="随机探索步数")
    parser.add_argument("--update_after", type=int, default=10_000, help="开始训练的最小步数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="经验回放池大小")
    parser.add_argument("--expl_noise_start", type=float, default=0.25, help="初始探索噪声")
    parser.add_argument("--expl_noise_end", type=float, default=0.05, help="最终探索噪声")
    parser.add_argument("--noise_decay_steps", type=int, default=150000, help="噪声衰减步数")
    parser.add_argument("--eval_interval", type=int, default=5_000, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估时的 episode 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 运行训练
    agent, log_rewards = train(args)

    # 渲染演示
    print("\n--- 开始仿真演示 ---")
    render_env = PhysicsDroneWrapper(HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM, 
        gui=True
    ))
    
    for i in range(5):
        state, _ = render_env.reset()
        done = False
        episode_reward = 0.0
        episode_step = 0
        
        while not done:
            reset_hidden = (episode_step == 0)
            action = agent.select_action(state, noise_std=0.0, reset_hidden=reset_hidden)
            
            state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_step += 1
        
        print(f"演示回合 {i+1} 结束，总奖励: {episode_reward:.2f}")
    
    render_env.close()
    print("演示结束。")
