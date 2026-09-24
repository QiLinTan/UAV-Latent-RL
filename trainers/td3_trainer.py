import numpy as np
import torch

from data.replay_buffer import ReplayBuffer
from data.stratified_replay_buffer import DifficultyStratifiedReplayBuffer
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

        success_replay_fraction = float(
            np.clip(getattr(args, "success_replay_fraction", 0.0), 0.0, 1.0)
        )
        self.success_replay_fraction = success_replay_fraction
        self.success_buffer = None
        if (
            success_replay_fraction > 0.0
            and bool(getattr(agent, "uses_success_replay", False))
            and not getattr(agent, "uses_context_replay", False)
        ):
            if bool(getattr(args, "success_replay_stratified", False)):
                self.success_buffer = DifficultyStratifiedReplayBuffer(
                    state_dim,
                    action_dim,
                    max_size=int(getattr(args, "success_replay_size", 50_000)),
                    difficulties=getattr(
                        args, "success_replay_difficulties", (1, 2, 3, 5)
                    ),
                )
            else:
                self.success_buffer = ReplayBuffer(
                    state_dim,
                    action_dim,
                    max_size=int(getattr(args, "success_replay_size", 50_000)),
                )
        self._episode_success_transitions = []

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
        self.request_stop = False
        self.stop_reason = None

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
        if hasattr(self.agent, "project_action"):
            action = self.agent.project_action(action)
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
            if self.success_buffer is not None:
                self._episode_success_transitions.append(
                    (
                        self.state.copy(),
                        np.asarray(action, dtype=np.float32).copy(),
                        float(scaled_reward),
                        next_state.copy(),
                        bool(done),
                    )
                )

        info = dict(info or {})
        replay_size = int(self.buffer.size)
        if hasattr(self.env, "set_replay_size"):
            self.env.set_replay_size(replay_size)
        info["replay_size"] = replay_size

        self.last_action = action

        self.state = next_state
        self.episode_return += reward
        self.episode_step += 1

        if done:
            if self.success_buffer is not None:
                if bool(info.get("success", False)):
                    for transition in self._episode_success_transitions:
                        if isinstance(
                            self.success_buffer, DifficultyStratifiedReplayBuffer
                        ):
                            self.success_buffer.push(
                                int(info.get("route_blocking_tree_count", 0)),
                                *transition,
                            )
                        else:
                            self.success_buffer.push(*transition)
                self._episode_success_transitions.clear()
            if hasattr(self.env, "record_episode_outcome"):
                self.env.record_episode_outcome(
                    success=bool(info.get("success", False)),
                    collision=bool(info.get("collision", False)),
                    done_reason=info.get("done_reason"),
                )
            if hasattr(self.env, "get_curriculum_metrics"):
                info.update(self.env.get_curriculum_metrics())
            info["replay_size"] = replay_size
            self.last_info = info
            # ✅ episode结束回调
            for cb in self.callbacks:
                cb.on_episode_end(self)

            self.state, _ = self.env.reset()
            self.state = preprocess_state(self.state.reshape(-1))
            if hasattr(self.agent, "reset_episode"):
                self.agent.reset_episode()
            self.episode_return = 0
            self.episode_step = 0
        else:
            self.last_info = info

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
            if self.success_buffer is not None:
                train_info = self.agent.train(
                    self.buffer,
                    batch_size=args.batch_size,
                    success_replay_buffer=self.success_buffer,
                    success_replay_fraction=self.success_replay_fraction,
                )
                if isinstance(
                    self.success_buffer, DifficultyStratifiedReplayBuffer
                ):
                    train_info.update(
                        {
                            f"success_replay/level_{level}_size": float(size)
                            for level, size in self.success_buffer.strata_sizes.items()
                        }
                    )
                    train_info.update(
                        {
                            f"success_replay/level_{level}_sample_count": float(count)
                            for level, count in self.success_buffer.last_sample_counts.items()
                        }
                    )
                return train_info
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

            if self.request_stop:
                print(f"[Train stop] {self.stop_reason or 'requested by callback'}")
                break

            if t % 1000 == 0:
                print(f"[Step {t}] buffer={self.buffer.size}")

        for cb in self.callbacks:
            cb.on_train_end(self)
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
