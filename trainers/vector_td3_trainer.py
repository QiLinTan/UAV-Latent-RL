from __future__ import annotations

from numbers import Real

import numpy as np

from data.replay_buffer import ReplayBuffer
from trainers.callbacks.noise import NoiseScheduler


class VectorTD3Trainer:
    """Off-policy TD3 collector for an autoresetting batched environment."""

    def __init__(self, env, agent, args):
        if not getattr(env, "autoreset", False):
            raise ValueError("VectorTD3Trainer currently requires an autoresetting environment.")

        self.env = env
        self.agent = agent
        self.args = args
        self.num_envs = int(env.num_envs)
        self.state_dim = int(env.observation_space.shape[-1])
        self.action_dim = int(env.action_space.shape[-1])
        self.max_action = 1.0
        self.buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            max_size=args.buffer_size,
        )
        self.state, _ = self.env.reset(seed=args.seed)
        self.state = self._validate_state(self.state)

        self.noise_scheduler = NoiseScheduler(
            start=args.expl_noise_start,
            end=args.expl_noise_end,
            decay_steps=args.noise_decay_steps,
        )
        self.updates_per_transition = float(args.updates_per_transition)
        self._update_budget = 0.0

        self.total_steps = 0
        self.vector_steps = 0
        self.next_progress_step = int(args.progress_interval)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_return = 0.0
        self.episode_step = 0
        self.train_steps_this_tick = 0
        self.callbacks = []
        self.last_action = None
        self.last_info = {}
        self.last_train_info = None
        self.last_eval_info = None
        self.last_eval_step = None

    def _validate_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        expected = (self.num_envs, self.state_dim)
        if state.shape != expected:
            raise ValueError(f"Expected batched state shape {expected}, got {state.shape}.")
        if not np.isfinite(state).all():
            raise FloatingPointError("Environment observation contains NaN or Inf.")
        return state

    def _select_actions(self):
        args = self.args
        if self.total_steps < args.start_timesteps:
            actions = self.env.sample_actions()
        else:
            if hasattr(self.agent, "select_actions"):
                actions = self.agent.select_actions(self.state)
            else:
                actions = np.stack(
                    [self.agent.select_action(state) for state in self.state],
                    axis=0,
                )
            noise = self.noise_scheduler.get_noise(self.total_steps)
            actions = actions + noise * np.random.randn(*actions.shape)
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f"Expected policy action shape {expected}, got {actions.shape}.")
        if not np.isfinite(actions).all():
            raise FloatingPointError("Policy action contains NaN or Inf.")
        return np.clip(actions, -self.max_action, self.max_action)

    @staticmethod
    def _mean_numeric_info(infos):
        keys = set().union(*(info.keys() for info in infos))
        summary = {}
        for key in keys:
            values = [info.get(key) for info in infos]
            if all(isinstance(value, (Real, np.number, bool)) for value in values):
                summary[key] = float(np.mean(values))
        return summary

    def _notify_episode_end(self, index, info):
        self.episode_return = float(self.episode_returns[index])
        self.episode_step = int(self.episode_steps[index])
        self.last_info = dict(info)
        for callback in self.callbacks:
            callback.on_episode_end(self)
        self.episode_returns[index] = 0.0
        self.episode_steps[index] = 0

    def step_env(self):
        actions = self._select_actions()
        next_state, reward, terminated, truncated, infos = self.env.step(actions)
        next_state = self._validate_state(next_state)
        reward = np.asarray(reward, dtype=np.float32).reshape(self.num_envs)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(self.num_envs)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(self.num_envs)
        done = terminated | truncated
        if not np.isfinite(reward).all():
            raise FloatingPointError("Environment reward contains NaN or Inf.")

        scaled_reward = reward * float(self.args.reward_scale)
        self.buffer.push_batch(
            self.state,
            actions,
            scaled_reward,
            next_state,
            done,
        )

        self.last_action = actions
        self.last_info = self._mean_numeric_info(infos)
        self.state = next_state
        self.episode_returns += reward
        self.episode_steps += 1

        for index in np.flatnonzero(done):
            self._notify_episode_end(index, infos[index])

    def train_step(self):
        args = self.args
        if self.total_steps < args.update_after or self.buffer.size < args.batch_size:
            return None
        if self.vector_steps % args.train_every != 0:
            return None

        self._update_budget += self.num_envs * self.updates_per_transition
        latest_info = None
        while self._update_budget >= 1.0:
            if hasattr(self.agent, "set_env_step"):
                self.agent.set_env_step(self.total_steps)
            latest_info = self.agent.train(self.buffer, batch_size=args.batch_size)
            self._update_budget -= 1.0
            self.train_steps_this_tick += 1
        return latest_info

    def run(self):
        args = self.args
        while self.total_steps < args.total_steps:
            self.step_env()
            self.vector_steps += 1
            self.total_steps += self.num_envs
            self.last_train_info = self.train_step()

            for callback in self.callbacks:
                callback.on_step(self)

            if args.progress_interval > 0 and self.total_steps >= self.next_progress_step:
                print(
                    f"[Transitions {self.total_steps}] vector_steps={self.vector_steps} "
                    f"buffer={self.buffer.size} updates={self.train_steps_this_tick}"
                )
                while self.next_progress_step <= self.total_steps:
                    self.next_progress_step += args.progress_interval

        for callback in self.callbacks:
            callback.on_train_end(self)

    def add_callback(self, callback):
        self.callbacks.append(callback)
