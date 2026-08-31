import numpy as np
import torch

from data.replay_buffer import ReplayBuffer
from data.hierarchical_replay_buffer import HierarchicalReplayBuffer
from envs.preprocess import preprocess_state
from trainers.callbacks.noise import NoiseScheduler


class TD3Trainer:
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.args = args

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.shape[-1])
        action_scale = float(np.clip(getattr(args, "action_scale", 1.0), 0.0, 1.0))
        self.max_action = float(env.action_space.high.flatten()[0]) * action_scale

        if getattr(agent, "uses_context_replay", False):
            self.buffer = HierarchicalReplayBuffer(
                state_dim,
                action_dim,
                context_dim=agent.context_dim,
                max_size=args.buffer_size,
            )
        else:
            self.buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

        self.state, _ = self.env.reset(seed=args.seed)
        self.state = preprocess_state(self.state.reshape(-1))
        if hasattr(self.agent, "reset_episode"):
            self.agent.reset_episode()
        if hasattr(self.agent, "configure_motor_action_interface"):
            self.agent.configure_motor_action_interface(self.env)

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
        uses_context_replay = bool(getattr(self.agent, "uses_context_replay", False))
        context = None
        if uses_context_replay:
            context = self.agent.prepare_runtime_context(self.state)

        teacher_timesteps = int(getattr(args, "teacher_timesteps", 0))
        teacher_supervision_timesteps = int(
            getattr(args, "teacher_supervision_timesteps", teacher_timesteps)
        )
        if teacher_supervision_timesteps <= 0:
            teacher_supervision_timesteps = teacher_timesteps
        teacher_action = None
        if (
            uses_context_replay
            and self.total_steps <= teacher_supervision_timesteps
            and hasattr(self.agent, "teacher_action_from_env")
        ):
            teacher_action = self.agent.teacher_action_from_env(self.env)
            if hasattr(self.agent, "supervised_actor_step"):
                self.agent.supervised_actor_step(
                    context,
                    teacher_action,
                    updates=int(getattr(args, "teacher_bc_updates_per_step", 1)),
                )
        using_teacher = (
            teacher_action is not None and self.total_steps <= teacher_timesteps
        )

        # A conventional controller supplies a stable warm start for the
        # low-level tracking-only experiment. It is not used by the final actor.
        if using_teacher:
            teacher_noise = float(getattr(args, "teacher_exploration_noise", 0.0))
            if teacher_noise > 0.0 and hasattr(self.agent, "add_safe_exploration_noise"):
                action = self.agent.add_safe_exploration_noise(
                    teacher_action,
                    teacher_noise,
                )
            else:
                action = teacher_action
        elif self.total_steps < args.start_timesteps:
            if uses_context_replay and hasattr(self.agent, "sample_safe_random_action"):
                action = self.agent.sample_safe_random_action()
            else:
                action = np.random.uniform(
                    -self.max_action,
                    self.max_action,
                    size=(self.env.action_space.shape[-1],),
                )
        else:
            if uses_context_replay:
                action = self.agent.action_from_context(context)
            else:
                action = self.agent.select_action(self.state)

            noise = self.noise_scheduler.get_noise(self.total_steps)
            if uses_context_replay and hasattr(self.agent, "add_safe_exploration_noise"):
                action = self.agent.add_safe_exploration_noise(action, noise)
            else:
                action = action + noise * np.random.randn(*action.shape)

        action = np.clip(action, -self.max_action, self.max_action)
        if uses_context_replay:
            self.agent.record_executed_action(action)

        next_obs, reward, terminated, truncated, info = self.env.step(action.reshape(1, -1))
        done = terminated or truncated

        next_state = preprocess_state(next_obs.reshape(-1))

        if uses_context_replay:
            self.agent.advance_runtime_step()
            next_context = self.agent.prepare_runtime_context(next_state)
            training_reward = float(reward)
            if hasattr(self.agent, "compute_training_reward"):
                training_reward = self.agent.compute_training_reward(
                    context=context,
                    next_context=next_context,
                    next_state=next_state,
                    environment_reward=reward,
                    done=done,
                    info=info,
                )
            scaled_reward = training_reward * float(args.reward_scale)
            self.buffer.push(
                self.state,
                action,
                scaled_reward,
                next_state,
                done,
                context=context,
                next_context=next_context,
            )
        else:
            scaled_reward = float(reward) * float(args.reward_scale)
            self.buffer.push(self.state, action, scaled_reward, next_state, done)

        self.last_action = action
        self.last_info = info  # 保存环境 info

        self.state = next_state
        self.episode_return += reward
        self.episode_step += 1

        if done:
            if hasattr(self.env, "record_episode_outcome"):
                self.env.record_episode_outcome(bool(info.get("success", False)))
            # ✅ episode结束回调
            for cb in self.callbacks:
                cb.on_episode_end(self)

            self.state, _ = self.env.reset()
            self.state = preprocess_state(self.state.reshape(-1))
            if hasattr(self.agent, "reset_episode"):
                self.agent.reset_episode()
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
            if hasattr(self.agent, "set_env_step"):
                self.agent.set_env_step(self.total_steps)
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
