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
from collections import deque
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
        assert state.shape[-1] == 72, f"Critic state 维度错误: {state.shape[-1]}，应该是72"
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
# Replay Buffer：标准随机采样（简化版，无序列采样）
# =============================================================

class SimpleReplayBuffer:
    """
    标准经验回放池 - 随机采样
    
    关键特性：
    - 存储完整的 (s, a_raw, r, s', done) transitions
    - 支持随机采样（无序列，无 burn-in）
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
        assert np.isscalar(reward) or reward.shape == () or reward.shape == (1,), f"Reward shape error: {np.shape(reward)}"
        
        idx = self.ptr % self.capacity
        self.state_buf[idx] = np.asarray(state)
        self.action_buf[idx] = np.asarray(action)
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = np.asarray(next_state)
        self.done_buf[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        随机采样一个 batch
        
        返回:
            - state: [batch_size, state_dim]
            - action: [batch_size, action_dim]
            - reward: [batch_size, 1]
            - next_state: [batch_size, state_dim]
            - done: [batch_size, 1]
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch_size {batch_size}")
        
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        return {
            "state": torch.as_tensor(self.state_buf[indices], device=device, dtype=torch.float32),
            "action": torch.as_tensor(self.action_buf[indices], device=device, dtype=torch.float32),
            "reward": torch.as_tensor(self.reward_buf[indices], device=device, dtype=torch.float32),
            "next_state": torch.as_tensor(self.next_state_buf[indices], device=device, dtype=torch.float32),
            "done": torch.as_tensor(self.done_buf[indices], device=device, dtype=torch.float32),
        }


# =============================================================
# 简单奖励计算辅助函数
# =============================================================

def compute_reward(obs, action, last_action):
    """
    计算奖励（针对 HoverAviary 原生观测）
    obs: 12维 KIN 观测 [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]
    """
    pos = obs[:3]
    rpy = obs[3:6]
    target = np.array([0, 0, 1.0])
    
    r_pos = -1.0 * np.linalg.norm(pos - target)**2
    r_att = -0.1 * np.linalg.norm(rpy)**2
    r_jerk = -0.05 * np.linalg.norm(action - last_action)**2
    r_survive = 0.1 if pos[2] > 0.1 and np.abs(rpy[0]) < 1.0 and np.abs(rpy[1]) < 1.0 else 0.0
    
    return float(r_pos + r_att + r_jerk + r_survive)


# =============================================================
# TD3 超参数配置（UAV 版本）
# =============================================================

@dataclass
class TD3UAVConfig:
    state_dim: int
    action_dim: int
    action_max: float
    
    # 状态拆分参数
    high_state_dim: int = 60  # 历史序列
    low_state_dim: int = 12   # 当前运动学状态
    enable_state_split: bool = True  # 启用状态语义拆分
    
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
    
    # Latent 启用开关
    use_latent: bool = True        # 是否使用 GRU 提取潜变量 z
    
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
        if self.config.use_latent:
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
            self.encoder_opt = torch.optim.Adam(self.actor_encoder.parameters(), lr=config.actor_lr)
        else:
            self.actor_encoder = self.actor_encoder_target = self.encoder_opt = None

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
        assert state.shape[0] == 72, f"select_action 输入state维度错误: {state.shape}"
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
        if not self.config.use_latent:
            # 🟢 禁用后：z = constant (0)
            self.current_latent = torch.zeros(1, self.config.latent_dim, device=self.device)
            self.debug_info['encoder_input_shigh'] = "Bypassed (use_latent=False)"
            self.debug_info['encoder_output_latent'] = self.current_latent.cpu().numpy()
            self.step_count += 1
        elif self.step_count % self.config.encoder_interval == 0 or self.current_latent is None:
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

    def train_step(self, replay_buffer: SimpleReplayBuffer, batch_size: int):
        """
        标准 TD3 训练步骤（简化版，无 Burn-in 序列）
        """
        if replay_buffer.size < batch_size:
            return {}
        
        # 采样 batch
        batch = replay_buffer.sample(batch_size, self.device)
        
        states = batch["state"]      # [batch_size, 72]
        actions = batch["action"]    # [batch_size, 4]
        rewards = batch["reward"]    # [batch_size, 1]
        next_states = batch["next_state"]  # [batch_size, 72]
        dones = batch["done"]        # [batch_size, 1]
        
        info = {}
        
        # ========== Critic 更新 ==========
        with torch.no_grad():
            # 拆分 next_state 为 s_high 和 s_low
            next_s_low = next_states[:, :12]
            next_s_high = next_states[:, 12:]
            
            # 计算 target action（使用 target network）
            if self.config.use_latent and next_s_high.shape[-1] > 0:
                next_latent, _ = self.actor_encoder_target(next_s_high, None)
            else:
                next_latent = torch.zeros(batch_size, self.config.latent_dim, device=self.device)
            
            next_action = self.actor_decoder_target(next_latent, next_s_low)
            
            # TD3 policy smoothing
            noise = (torch.randn_like(next_action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (next_action + noise).clamp(-self.config.action_max, self.config.action_max)
            
            # 计算 target Q
            target_q1 = self.critic1_target(next_states, next_action)
            target_q2 = self.critic2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.config.gamma * target_q
        
        # 计算 current Q
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        # 更新 Critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()
        
        info["critic_loss"] = float(critic_loss.item())
        try:
            info["q_mean"] = float(torch.cat([q1, q2], dim=0).mean().item())
        except:
            info["q_mean"] = None
        
        # ========== Actor 更新（Delayed） ==========
        if self.total_it % self.config.policy_delay == 0:
            # 拆分 state 为 s_high 和 s_low
            s_low = states[:, :12]
            s_high = states[:, 12:]
            
            # 计算 latent
            if self.config.use_latent and s_high.shape[-1] > 0:
                latent, _ = self.actor_encoder(s_high, None)
            else:
                latent = torch.zeros(batch_size, self.config.latent_dim, device=self.device)
            
            # 计算动作
            current_action = self.actor_decoder(latent, s_low)
            
            # Actor loss：最大化 Q（所以最小化负 Q）
            actor_loss = -self.critic1(states, current_action).mean()
            
            # 更新 Actor
            if self.config.use_latent:
                self.encoder_opt.zero_grad()
            self.decoder_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ([p for p in self.actor_encoder.parameters()] if self.config.use_latent else []) +
                list(self.actor_decoder.parameters()), 1.0
            )
            if self.config.use_latent:
                self.encoder_opt.step()
            self.decoder_opt.step()
            
            # 软更新 target 网络
            if self.config.use_latent:
                self._soft_update(self.actor_encoder_target, self.actor_encoder)
            self._soft_update(self.actor_decoder_target, self.actor_decoder)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
            
            info["actor_loss"] = float(actor_loss.item())
        
        self.total_it += 1
        return info

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
    for ep in range(episodes):
        raw_obs, _ = env.reset()
        action_deque = deque(maxlen=15)
        for _ in range(15):
            action_deque.append(np.zeros(4))
        
        state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
        done = False
        ep_ret = 0.0
        episode_step = 0
        last_action = np.zeros(4)
        
        while not done:
            reset_hidden = (episode_step == 0)
            action = agent.select_action(state, noise_std=noise, reset_hidden=reset_hidden)
            
            raw_obs, reward, terminated, truncated, _ = env.step(action.reshape(1, -1))
            done = terminated or truncated
            
            # 维护历史
            action_deque.append(action)
            state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
            
            # 计算奖励
            reward = compute_reward(raw_obs, action, last_action)
            last_action = action.copy()
            
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
    rng = np.random.default_rng(args.seed)
    # 标准初始化方式：在创建 env 时传入 initial_xyzs / initial_rpys
    # 空中初始化（更推荐随机高度：1.0~1.5m），并通过 seed 保证可复现
    INIT_XYZS = np.array([[0.0, 0.0, rng.uniform(1.0, 1.5)]], dtype=np.float32)
    # 保证飞机姿态正确（r,p,y in radians）
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # 直接使用原生 HoverAviary，无包装
    env = HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        gui=False,
        pyb_freq=240,
        ctrl_freq=120
    )
    eval_env = HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        gui=True,
        pyb_freq=240,
        ctrl_freq=120
    )

    writer = SummaryWriter(log_dir=f"runs/TD3_Latent_UAV_{args.seed}")

    state_dim = env.observation_space.shape[0]
    assert state_dim == 72, f"Expected state_dim==72 but got {state_dim}. Ensure env observation space matches processed obs."
    print(f"Corrected state_dim: {state_dim}") # 通常应该是 72
    action_dim = env.action_space.shape[0]
    # action_max = float(env.action_space.high[0]) # 原始是 1.0
    action_max = 1.0 # 标准化动作范围 [-1, 1]（由 tanh 输出）

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    config = TD3UAVConfig(
        state_dim=72,            # 总维度
        action_dim=action_dim,
        action_max=action_max,
        high_state_dim=60,       # 历史序列
        low_state_dim=12,        # 当前运动学
        enable_state_split=True, # 必须开启
        latent_dim=args.latent_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        hidden_dim=args.hidden_dim,
        encoder_interval=args.encoder_interval,
        burn_in_length=args.burn_in_length,
        learning_length=args.learning_length,
        enable_burn_in=True,     # 建议开启，GRU 才有意义
        actor_lr=args.actor_lr,
        use_latent=args.use_latent,
        critic_lr=args.critic_lr,
        tau=args.tau,
        policy_delay=args.policy_delay,
    )
    
    agent = TD3LatentUAV(config, device)
    buffer = SimpleReplayBuffer(state_dim, action_dim, capacity=args.buffer_size)

    # 初始化 KIN 观测处理
    raw_obs, _ = env.reset(seed=args.seed)  # 原始 12 维 KIN 观测
    action_deque = deque(maxlen=15)  # 维护 15 步历史
    for _ in range(15):
        action_deque.append(np.zeros(4))
    
    # 构造初始状态（12 维 obs + 60 维历史 = 72 维）
    state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    episode_reward = 0.0
    episode_step = 0
    log_rewards = []
    info = None
    last_eval_ret = None
    last_train_info = None
    last_action = np.zeros(4)

    try:
        for t in range(1, args.total_steps + 1):
            # 计算噪声衰减
            expl_noise = max(
                args.expl_noise_end,
                args.expl_noise_start - (t / args.noise_decay_steps) * (args.expl_noise_start - args.expl_noise_end)
            )

            if t < args.start_steps:
                # 随机探索
                action = np.random.uniform(-action_max, action_max, size=(action_dim,))
            else:
                reset_hidden = (episode_step == 0)
                action = agent.select_action(state, noise_std=expl_noise, reset_hidden=reset_hidden)

            # 执行环境步进（HoverAviary 原生输出 12 维 KIN 观测）
            raw_obs, reward, terminated, truncated, _ = env.step(action.reshape(1, -1))
            done = terminated or truncated
            
            # 维护历史动作队列
            action_deque.append(action)
            
            # 构造 72 维状态：12 维 obs + 60 维历史
            next_state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
            
            # 计算奖励（简化版本）
            reward = compute_reward(raw_obs, action, last_action)
            
            # 存储到 buffer
            buffer.push(state, action, reward, next_state, float(done))

            if t % 1000 == 0:
                print(f"Step {t}: Buffer size = {buffer.size}, Reward = {reward:.3f}, Action = {action}")

            state = next_state
            episode_reward += reward
            episode_step += 1
            last_action = action.copy()

            # 训练：每步都训练（如果 buffer 足够）
            if t >= args.update_after and buffer.size >= args.batch_size:
                print(f"TRAIN STEP CALLED @ Step {t}")  # 调试：验证训练是否真的被触发
                info = agent.train_step(buffer, args.batch_size)
                if info:
                    last_train_info = info
           
            # 处理 Episode 结束
            if done:
                writer.add_scalar("Reward/Train_Episode_Reward", episode_reward, t)
                log_rewards.append(episode_reward)
                print(f"Episode end @ step {t}: Total Reward = {episode_reward:.2f}")
                
                raw_obs, _ = env.reset()
                action_deque.clear()
                for _ in range(15):
                    action_deque.append(np.zeros(4))
                state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
                episode_reward = 0.0
                episode_step = 0

            # 评估
            if t % args.eval_interval == 0:
                avg_eval_ret = evaluate(eval_env, agent, episodes=args.eval_episodes)
                last_eval_ret = avg_eval_ret
                writer.add_scalar("Reward/Eval_Average_Return", avg_eval_ret, t)
                
                # 写入训练统计
                if last_train_info:
                    for key, value in last_train_info.items():
                        if value is not None:
                            writer.add_scalar(f"Loss/{key}", value, t)
                writer.add_scalar("Stats/Exploration_Noise", expl_noise, t)
                print(f"Evaluation @ step {t}: Avg Return = {avg_eval_ret:.2f}")
                
    except KeyboardInterrupt:
        print("\n检测到用户中断...")

    writer.close()
    return agent, log_rewards


# =============================================================
# 命令行参数解析
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TD3-Latent UAV (Hierarchical Control) for HoverAviary")
    
    # 状态拆分参数
    parser.add_argument("--high_state_dim", type=int, default=60, help="s_high 维度 (历史序列)")
    parser.add_argument("--low_state_dim", type=int, default=12, help="s_low 维度 (当前运动学)")
    parser.add_argument("--enable_state_split", action="store_true", default=True, help="启用状态语义拆分（默认启用）")
    parser.add_argument("--disable_state_split", dest="enable_state_split", action="store_false", help="禁用状态语义拆分")
    
    # Latent 开关
    parser.add_argument("--use_latent", action="store_true", default=True, help="是否使用 latent GRU (默认启用)")
    parser.add_argument("--no_latent", dest="use_latent", action="store_false", help="禁用 latent GRU")
    
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
    parser.add_argument("--eval_interval", type=int, default=20_000, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=3, help="评估时的 episode 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 运行训练
    agent, log_rewards = train(args)

    # 渲染演示
    print("\n--- 开始仿真演示 ---")
    render_env = HoverAviary(
        obs=ObservationType.KIN, 
        act=ActionType.RPM, 
        initial_xyzs=np.array([[0.0, 0.0, 1.2]], dtype=np.float32),
        initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        gui=True
    )
    
    for i in range(5):
        raw_obs, _ = render_env.reset()
        action_deque = deque(maxlen=15)
        for _ in range(15):
            action_deque.append(np.zeros(4))
        
        state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
        done = False
        episode_reward = 0.0
        episode_step = 0
        last_action = np.zeros(4)
        
        while not done:
            reset_hidden = (episode_step == 0)
            action = agent.select_action(state, noise_std=0.0, reset_hidden=reset_hidden)
            
            raw_obs, _, terminated, truncated, _ = render_env.step(action.reshape(1, -1))
            done = terminated or truncated
            
            # 维护历史
            action_deque.append(action)
            state = np.concatenate([raw_obs, np.array(action_deque).flatten()]).astype(np.float32)
            
            # 计算奖励
            reward = compute_reward(raw_obs, action, last_action)
            last_action = action.copy()
            
            episode_reward += reward
            episode_step += 1
        
        print(f"演示回合 {i+1} 结束，总奖励: {episode_reward:.2f}")
    
    render_env.close()
    print("演示结束。")
