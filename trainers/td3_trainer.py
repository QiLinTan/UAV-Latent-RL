import numpy as np
import torch

from data.replay_buffer import ReplayBuffer
from envs.preprocess import preprocess_state
from trainers.callbacks.noise import NoiseScheduler


class TD3Trainer:
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.args = args

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.shape[-1])
        self.max_action = float(env.action_space.high.flatten()[0])

        self.buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

        self.state, _ = self.env.reset(seed=args.seed)
        self.state = preprocess_state(self.state.reshape(-1))

        self.noise_scheduler = NoiseScheduler(
            start=args.expl_noise_start,
            end=args.expl_noise_end,
            decay_steps=args.noise_decay_steps,
        )

        self.total_steps = 0
        self.episode_return = 0
        self.episode_step = 0
        self.train_steps_this_tick = 0
        self.callbacks = []
        
        # 初始化日志相关属性
        self.last_action = None
        self.last_info = {}
        self.last_train_info = None

    def step_env(self):
        args = self.args

        # exploration
        if self.total_steps < args.start_timesteps:
            action = np.random.uniform(
                -self.max_action,
                self.max_action,
                size=(self.env.action_space.shape[-1],),
            )
        else:
            action = self.agent.select_action(self.state)

            noise = self.noise_scheduler.get_noise(self.total_steps)
            action = action + noise * np.random.randn(*action.shape)

        action = np.clip(action, -self.max_action, self.max_action)

        next_obs, reward, terminated, truncated, info = self.env.step(action.reshape(1, -1))
        done = terminated or truncated
        scaled_reward = float(reward) * float(args.reward_scale)

        next_state = preprocess_state(next_obs.reshape(-1))

        self.buffer.push(self.state, action, scaled_reward, next_state, done)

        self.last_action = action
        self.last_info = info  # 保存环境 info

        self.state = next_state
        self.episode_return += reward
        self.episode_step += 1

        if done:
            # ✅ episode结束回调
            for cb in self.callbacks:
                cb.on_episode_end(self)

            self.state, _ = self.env.reset()
            self.state = preprocess_state(self.state.reshape(-1))
            self.episode_return = 0
            self.episode_step = 0

    def train_step(self):
        args = self.args

        if (
            self.total_steps >= args.update_after
            and self.buffer.size >= args.batch_size
            and self.total_steps % args.train_every == 0
        ):
            self.train_steps_this_tick += 1
            return self.agent.train(self.buffer, batch_size=args.batch_size)

        return None

    def run(self):
        args = self.args

        for t in range(1, args.total_steps + 1):
            self.total_steps = t

            # 在评估间隔开始时重置计数器
            if t % args.eval_interval == 1:
                self.train_steps_this_tick = 0

            self.step_env()

            train_info = self.train_step()
            self.last_train_info = train_info

            for cb in self.callbacks:
                cb.on_step(self)

            if t % 1000 == 0:
                print(f"[Step {t}] buffer={self.buffer.size}")

        for cb in self.callbacks:
            cb.on_train_end(self)
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
