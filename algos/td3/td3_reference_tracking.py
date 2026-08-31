from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from algos.td3.networks import Actor, Critic, ResidualMotorActor, StructuredMotorActor
from algos.td3.td3_latent_only import _grad_norm
from envs.observation_layout import KIN_DIM, ForestObservationLayout
from models.motor_action_codec import (
    ASYMMETRIC_RPM,
    MotorActionCodec,
    MotorPhysicalLimits,
    SUPPORTED_MOTOR_ACTION_CODECS,
)
from models.reference_packet import (
    ActuatorConstraintLayer,
    AsyncReferenceBuffer,
    DegradedHoverController,
    RuleReferenceGenerator,
    TrajectoryLimits,
)


class TD3ReferenceTracking:
    """Low-level-only TD3 driven by deterministic feasible reference packets."""

    _KIN_MEAN = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    _KIN_STD = np.array(
        [0.5, 0.5, 0.5, np.pi, np.pi, np.pi, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
        dtype=np.float32,
    )

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        *,
        target_position=(3.5, 0.0, 1.0),
        ctrl_freq=120,
        sequence_length=15,
        reference_horizon_seconds=1.0,
        high_level_interval=8,
        reference_mode="line",
        max_reference_speed=0.8,
        max_reference_acceleration=2.0,
        max_reference_vertical_speed=0.5,
        lookahead_points=3,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        grad_clip_norm=1.0,
        action_l2_weight=0.10,
        motor_balance_weight=0.10,
        action_delta_weight=0.10,
        normalize_actor_q=True,
        motor_collective_fraction=0.60,
        motor_differential_fraction=0.25,
        motor_max_delta=None,
        actor_structure="plain",
        lower_action_history_steps=4,
        environment_reward_weight=0.0,
        actor_rl_start_step=0,
        fallback_decay=0.15,
        motor_action_mode=ASYMMETRIC_RPM,
        min_allowed_rpm=0.0,
        motor_max_delta_rpm=None,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
        self.ctrl_freq = int(ctrl_freq)
        self.control_dt = 1.0 / float(self.ctrl_freq)
        self.sequence_length = int(sequence_length)
        self.reference_horizon_seconds = float(reference_horizon_seconds)
        self.high_level_interval = int(high_level_interval)
        self.lookahead_points = int(lookahead_points)
        self.lookahead_seconds = (
            self.lookahead_points
            * self.reference_horizon_seconds
            / max(1, self.sequence_length - 1)
        )
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.grad_clip_norm = float(grad_clip_norm)
        self.action_l2_weight = float(action_l2_weight)
        self.motor_balance_weight = float(motor_balance_weight)
        self.action_delta_weight = float(action_delta_weight)
        self.normalize_actor_q = bool(normalize_actor_q)
        self.environment_reward_weight = float(environment_reward_weight)
        self.actor_rl_start_step = int(actor_rl_start_step)
        self.layout = ForestObservationLayout.from_total_dim(
            self.state_dim,
            action_dim=self.action_dim,
        )
        available_history_steps = self.layout.action_history_dim // self.action_dim
        self.lower_action_history_steps = min(
            max(0, int(lower_action_history_steps)),
            available_history_steps,
        )
        self.lower_action_history_dim = self.lower_action_history_steps * self.action_dim
        if self.ctrl_freq <= 0 or self.high_level_interval <= 0:
            raise ValueError("ctrl_freq and high_level_interval must be positive.")

        limits = TrajectoryLimits(
            max_speed=float(max_reference_speed),
            max_acceleration=float(max_reference_acceleration),
            max_vertical_speed=float(max_reference_vertical_speed),
        )
        self.reference_generator = RuleReferenceGenerator(
            sequence_length=self.sequence_length,
            horizon_seconds=self.reference_horizon_seconds,
            limits=limits,
            mode=reference_mode,
        )
        self.reference_mode = reference_mode
        self.reference_anchor = None
        self.reference_buffer = AsyncReferenceBuffer()
        if motor_action_mode not in (*SUPPORTED_MOTOR_ACTION_CODECS, "legacy_projected"):
            raise ValueError(
                "motor_action_mode must be 'asymmetric_rpm', "
                "'asymmetric_thrust', or 'legacy_projected'."
            )
        self.motor_action_mode = str(motor_action_mode)
        self.min_allowed_rpm = float(min_allowed_rpm)
        self.motor_max_delta_rpm = (
            None if motor_max_delta_rpm is None else float(motor_max_delta_rpm)
        )
        self.motor_action_codec = None
        self.legacy_max_action = 0.75
        self.actuator_constraints = ActuatorConstraintLayer(
            self.action_dim,
            self.legacy_max_action,
            motor_collective_fraction,
            motor_differential_fraction,
            max_delta=motor_max_delta,
        )
        self.fallback_controller = DegradedHoverController(
            self.action_dim,
            decay=fallback_decay,
        )
        self.motor_collective_fraction = float(motor_collective_fraction)
        self.motor_differential_fraction = float(motor_differential_fraction)

        # kin + current (position/velocity error) + lookahead errors
        # + packet age/valid + previous motor command
        self.policy_input_dim = (
            KIN_DIM
            + 3
            + 3
            + 3
            + 3
            + 1
            + 1
            + self.lower_action_history_dim
            + self.action_dim
        )
        self.context_dim = self.policy_input_dim
        self.context_previous_action_start = self.context_dim - self.action_dim
        self.current_position_error_slice = slice(KIN_DIM, KIN_DIM + 3)
        self.current_velocity_error_slice = slice(KIN_DIM + 3, KIN_DIM + 6)
        self.lookahead_position_error_slice = slice(KIN_DIM + 6, KIN_DIM + 9)
        self.lookahead_velocity_error_slice = slice(KIN_DIM + 9, KIN_DIM + 12)
        self.uses_context_replay = True

        if actor_structure == "plain":
            self.actor = Actor(
                self.policy_input_dim,
                self.action_dim,
                self.max_action,
            ).to(self.device)
        elif actor_structure == "structured":
            self.actor = StructuredMotorActor(
                self.policy_input_dim,
                self.action_dim,
                self.max_action,
                collective_fraction=self.motor_collective_fraction,
                differential_fraction=self.motor_differential_fraction,
                virtual_headroom=1.6,
            ).to(self.device)
        elif actor_structure == "residual":
            self.actor = ResidualMotorActor(
                self.policy_input_dim,
                self.action_dim,
                self.max_action,
                collective_fraction=self.motor_collective_fraction,
                differential_fraction=self.motor_differential_fraction,
                differential_headroom=1.6,
            ).to(self.device)
        else:
            raise ValueError(
                "actor_structure must be 'plain', 'residual', or 'structured'."
            )
        self.actor_structure = actor_structure
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.policy_input_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.runtime_step = 0
        self.total_it = 0
        self.env_step = 0
        self.high_level_enabled = True
        self.last_runtime_info = {}
        self.last_reference_sample = None
        self.reference_anchor = None
        self.last_bc_loss = None
        self._teacher_controller = None
        self._teacher_contexts = deque(maxlen=100_000)
        self._teacher_actions = deque(maxlen=100_000)

    def set_env_step(self, step: int):
        self.env_step = int(step)

    def reset_episode(self):
        if self._teacher_controller is not None:
            self._teacher_controller.reset()
        self.reference_buffer.clear()
        self.previous_action.fill(0.0)
        self.runtime_step = 0
        self.last_runtime_info = {}
        self.last_reference_sample = None
        self.reference_anchor = None

    def set_high_level_enabled(self, enabled: bool):
        self.high_level_enabled = bool(enabled)

    @classmethod
    def denormalize_kinematics(cls, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return state[:KIN_DIM] * cls._KIN_STD + cls._KIN_MEAN

    def _refresh_reference(self, state):
        kin = self.denormalize_kinematics(state)
        now = self.runtime_step * self.control_dt
        if self.reference_anchor is None:
            self.reference_anchor = kin[:3].copy()
        reference_target = (
            self.reference_anchor
            if self.reference_mode == "hover"
            else self.target_position
        )
        packet = self.reference_generator.generate(
            position=kin[:3],
            velocity=kin[6:9],
            target_position=reference_target,
            t_gen=now,
            t_start=now,
            t_receive=now,
        )
        if not self.reference_buffer.publish(packet):
            raise RuntimeError("Rule generator produced a stale or out-of-order packet.")

    def prepare_runtime_context(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        now = self.runtime_step * self.control_dt
        must_refresh = (
            not self.reference_buffer.has_value
            or (
                self.runtime_step % self.high_level_interval == 0
                and abs(self.reference_buffer.packet.t_gen - now) > 1e-9
            )
        )
        if must_refresh and self.high_level_enabled:
            self._refresh_reference(state)

        if not self.reference_buffer.has_value:
            self.last_reference_sample = None
            self.last_runtime_info = {
                "async_cache_version": 0.0,
                "async_cache_age_seconds": 0.0,
                "async_cache_age_ratio": 1.0,
                "async_cache_valid": 0.0,
                "async_reference_index": 0.0,
                "async_high_level_refreshed": 0.0,
                "async_degraded_fallback": 1.0,
                "reference_tracking_mode": 1.0,
            }
            short_history = self._short_action_history(state)
            return np.concatenate(
                [
                    state[:KIN_DIM],
                    np.zeros(12, dtype=np.float32),
                    np.array([1.0, 0.0], dtype=np.float32),
                    short_history,
                    self.previous_action,
                ]
            ).astype(np.float32)

        sample = self.reference_buffer.sample(now, lookahead_seconds=self.lookahead_seconds)
        self.last_reference_sample = sample
        kin = self.denormalize_kinematics(state)
        position = kin[:3]
        velocity = kin[6:9]
        position_scale = self._KIN_STD[:3]
        velocity_scale = self._KIN_STD[6:9]
        current_position_error = (sample["current_position"] - position) / position_scale
        current_velocity_error = (sample["current_velocity"] - velocity) / velocity_scale
        lookahead_position_error = (sample["lookahead_position"] - position) / position_scale
        lookahead_velocity_error = (sample["lookahead_velocity"] - velocity) / velocity_scale
        valid_value = float(sample["valid"])
        short_history = self._short_action_history(state)
        context = np.concatenate(
            [
                state[:KIN_DIM],
                np.clip(current_position_error, -4.0, 4.0),
                np.clip(current_velocity_error, -4.0, 4.0),
                np.clip(lookahead_position_error, -4.0, 4.0),
                np.clip(lookahead_velocity_error, -4.0, 4.0),
                np.array([sample["age_ratio"], valid_value], dtype=np.float32),
                short_history,
                self.previous_action,
            ]
        ).astype(np.float32)
        self.last_runtime_info = {
            "async_cache_version": float(sample["version"]),
            "async_cache_age_seconds": float(sample["age_seconds"]),
            "async_cache_age_ratio": float(sample["age_ratio"]),
            "async_cache_valid": valid_value,
            "async_reference_index": float(sample["current_index"]),
            "async_high_level_refreshed": float(must_refresh and self.high_level_enabled),
            "async_degraded_fallback": float(not sample["valid"]),
            "reference_tracking_mode": 1.0,
        }
        return context

    def _short_action_history(self, state) -> np.ndarray:
        if self.lower_action_history_dim == 0:
            return np.zeros(0, dtype=np.float32)
        history = np.asarray(
            state[KIN_DIM : self.layout.goal_start],
            dtype=np.float32,
        )
        return history[-self.lower_action_history_dim :].copy()

    def action_from_context(self, context) -> np.ndarray:
        if not bool(self.last_runtime_info.get("async_cache_valid", 0.0)):
            return self.fallback_controller(self.previous_action)
        context_t = torch.as_tensor(
            context,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(context_t).cpu().numpy().reshape(-1)
        return self.constrain_motor_action(
            action,
            previous_action=self.previous_action,
        )

    def project_motor_action(self, action) -> np.ndarray:
        return self.constrain_motor_action(action)

    def configure_motor_action_interface(self, env):
        """Bind the selected normalized action semantics to the environment."""

        if self.motor_action_mode == "legacy_projected":
            self.motor_action_codec = None
            if hasattr(env, "set_motor_action_codec"):
                env.set_motor_action_codec(None)
            return None
        if self.max_action < 1.0 - 1e-6:
            raise ValueError(
                "The physical motor codecs require max_action=1.0; a narrower "
                "global action clip would destroy teacher control authority."
            )
        if not hasattr(env, "set_motor_action_codec"):
            raise TypeError(
                "The environment must expose set_motor_action_codec() for the "
                "new direct-motor interfaces."
            )
        limits = MotorPhysicalLimits(
            min_rpm=self.min_allowed_rpm,
            hover_rpm=float(env.HOVER_RPM),
            max_rpm=float(env.MAX_RPM),
            kf=float(env.KF),
        )
        self.motor_action_codec = MotorActionCodec(
            limits,
            mode=self.motor_action_mode,
            max_delta_rpm=self.motor_max_delta_rpm,
        )
        env.set_motor_action_codec(self.motor_action_codec)
        return self.motor_action_codec

    def constrain_motor_action(self, action, previous_action=None) -> np.ndarray:
        if self.motor_action_mode == "legacy_projected":
            return self.actuator_constraints(
                action,
                previous_action=previous_action,
            )
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.action_dim),
            -1.0,
            1.0,
        )

    def sample_safe_random_action(self) -> np.ndarray:
        return self.constrain_motor_action(
            np.random.uniform(-0.15, 0.15, size=self.action_dim)
        )

    def add_safe_exploration_noise(self, action, noise_std: float) -> np.ndarray:
        return self.constrain_motor_action(
            np.asarray(action, dtype=np.float32)
            + float(noise_std) * np.random.randn(self.action_dim)
        )

    def teacher_action_from_env(self, env) -> np.ndarray:
        """Return a geometric/PID motor command for warm-start supervision."""

        if self.motor_action_mode != "legacy_projected" and self.motor_action_codec is None:
            self.configure_motor_action_interface(env)
        if self.last_reference_sample is None:
            return self.fallback_controller(self.previous_action)
        if self._teacher_controller is None:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

            self._teacher_controller = DSLPIDControl(drone_model=env.DRONE_MODEL)
        drone_state = env._getDroneStateVector(0)
        rpm, _, _ = self._teacher_controller.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=drone_state[0:3],
            cur_quat=drone_state[3:7],
            cur_vel=drone_state[10:13],
            cur_ang_vel=drone_state[13:16],
            target_pos=self.last_reference_sample["lookahead_position"],
            target_vel=self.last_reference_sample["lookahead_velocity"],
        )
        if self.motor_action_mode == "legacy_projected":
            normalized = (
                np.asarray(rpm, dtype=np.float32) / float(env.HOVER_RPM) - 1.0
            ) / 0.05
            return self.actuator_constraints(normalized)
        return self.motor_action_codec.rpm_to_normalized_action(rpm).astype(np.float32)

    def supervised_actor_step(self, context, teacher_action, updates: int = 1) -> float:
        context_value = np.asarray(context, dtype=np.float32).reshape(self.context_dim).copy()
        action_value = np.asarray(teacher_action, dtype=np.float32).reshape(self.action_dim).copy()
        copies = 4 if bool(self.last_runtime_info.get("async_high_level_refreshed", 0.0)) else 1
        for _ in range(copies):
            self._teacher_contexts.append(context_value)
            self._teacher_actions.append(action_value)

        loss = None
        for _ in range(max(1, int(updates))):
            batch_size = min(256, len(self._teacher_contexts))
            indices = np.random.randint(0, len(self._teacher_contexts), size=batch_size)
            context_t = torch.as_tensor(
                np.stack([self._teacher_contexts[i] for i in indices]),
                dtype=torch.float32,
                device=self.device,
            )
            target_t = torch.as_tensor(
                np.stack([self._teacher_actions[i] for i in indices]),
                dtype=torch.float32,
                device=self.device,
            )
            prediction = self.actor(context_t)
            prediction_collective = prediction.mean(dim=1, keepdim=True)
            target_collective = target_t.mean(dim=1, keepdim=True)
            prediction_differential = prediction - prediction_collective
            target_differential = target_t - target_collective
            collective_loss = F.mse_loss(prediction_collective, target_collective)
            differential_loss = F.mse_loss(
                prediction_differential,
                target_differential,
            )
            loss = 20.0 * collective_loss + 5.0 * differential_loss
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.last_bc_loss = float(loss.item())
        return self.last_bc_loss

    def compute_training_reward(
        self,
        *,
        context,
        next_context,
        next_state,
        environment_reward,
        done,
        info,
    ) -> float:
        context = np.asarray(context, dtype=np.float32)
        next_context = np.asarray(next_context, dtype=np.float32)
        current_position_error = float(
            np.linalg.norm(context[self.lookahead_position_error_slice])
        )
        next_position_error = float(
            np.linalg.norm(next_context[self.lookahead_position_error_slice])
        )
        next_velocity_error = float(
            np.linalg.norm(next_context[self.lookahead_velocity_error_slice])
        )
        next_state = np.asarray(next_state, dtype=np.float32)
        attitude = float(np.linalg.norm(next_state[3:5]))
        angular_velocity = float(np.linalg.norm(next_state[9:12]))
        progress = current_position_error - next_position_error
        reward = (
            2.0 * progress
            + 0.5 * np.exp(-2.0 * next_position_error)
            - 0.25 * next_velocity_error
            - 0.15 * attitude
            - 0.05 * angular_velocity
            + self.environment_reward_weight * float(environment_reward)
        )
        if done:
            reason = str((info or {}).get("done_reason", "unknown"))
            if reason in {"attitude_bound", "height_bound", "collision", "xy_bound"}:
                reward -= 10.0
            elif reason == "success":
                reward += 10.0
        return float(reward)

    def record_executed_action(self, action):
        self.previous_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim).copy()

    def advance_runtime_step(self):
        self.runtime_step += 1

    def select_action(self, state):
        context = self.prepare_runtime_context(state)
        action = self.action_from_context(context)
        self.record_executed_action(action)
        self.advance_runtime_step()
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        _, action, _, reward, not_done, context, next_context = replay_buffer.sample(batch_size)
        action = action.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)
        context = context.to(self.device)
        next_context = next_context.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip,
                self.noise_clip,
            )
            next_action = (self.actor_target(next_context) + noise).clamp(
                -self.max_action,
                self.max_action,
            )
            target_q1, target_q2 = self.critic_target(next_context, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(context, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = _grad_norm(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        actor_updated = False
        actor_loss_value = None
        actor_sat_pct = None
        actor_grad_norm = 0.0
        action_l2_loss = motor_balance_loss = action_delta_loss = actor_q_scale = None
        if (
            self.env_step > self.actor_rl_start_step
            and self.total_it % self.policy_freq == 0
        ):
            actor_actions = self.actor(context)
            previous_action = context[:, self.context_previous_action_start :]
            action_l2_loss = actor_actions.pow(2).mean()
            motor_balance_loss = actor_actions.var(dim=1, unbiased=False).mean()
            action_delta_loss = (actor_actions - previous_action).pow(2).mean()
            actor_q = self.critic.Q1(context, actor_actions)
            actor_q_scale = (
                actor_q.detach().abs().mean().clamp_min(1.0)
                if self.normalize_actor_q
                else actor_q.new_tensor(1.0)
            )
            actor_loss = (
                -actor_q.mean() / actor_q_scale
                + self.action_l2_weight * action_l2_loss
                + self.motor_balance_weight * motor_balance_loss
                + self.action_delta_weight * action_delta_loss
            )
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            actor_sat_pct = float(
                (actor_actions.detach().abs() >= self.max_action - 1e-3).float().mean().item()
            )
            actor_updated = True

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        def scalar(value):
            return None if value is None else float(value.detach().item())

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_loss": actor_loss_value,
            "actor_grad_norm": float(actor_grad_norm),
            "actor_updated": actor_updated,
            "actor_sat_pct": actor_sat_pct,
            "action_l2_loss": scalar(action_l2_loss),
            "motor_balance_loss": scalar(motor_balance_loss),
            "action_delta_loss": scalar(action_delta_loss),
            "actor_q_scale": scalar(actor_q_scale),
            "behavior_clone_loss": self.last_bc_loss,
            "q1_mean": float(current_q1.detach().mean().item()),
            "q2_mean": float(current_q2.detach().mean().item()),
            "q_target_mean": float(target_q.detach().mean().item()),
            "q_gap_abs_mean": float(
                (current_q1.detach() - current_q2.detach()).abs().mean().item()
            ),
            "hierarchical_async": 1.0,
            "reference_tracking_mode": 1.0,
            "reference_sequence_length": float(self.sequence_length),
            "high_level_interval": float(self.high_level_interval),
            "reference_horizon_seconds": float(self.reference_horizon_seconds),
            "actor_rl_start_step": float(self.actor_rl_start_step),
            "structured_motor_actor": float(self.actor_structure == "structured"),
            "lower_action_history_steps": float(self.lower_action_history_steps),
            "context_replay_enabled": 1.0,
            "motor_collective_fraction": float(self.motor_collective_fraction),
            "motor_differential_fraction": float(self.motor_differential_fraction),
            "env_step": float(self.env_step),
            **self.last_runtime_info,
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), f"{filename}_critic")
        torch.save(self.actor.state_dict(), f"{filename}_actor")
        torch.save(self.critic_optimizer.state_dict(), f"{filename}_critic_optimizer")
        torch.save(self.actor_optimizer.state_dict(), f"{filename}_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(f"{filename}_critic", map_location=self.device))
        self.actor.load_state_dict(torch.load(f"{filename}_actor", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(f"{filename}_critic_optimizer", map_location=self.device)
        )
        self.actor_optimizer.load_state_dict(
            torch.load(f"{filename}_actor_optimizer", map_location=self.device)
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.reset_episode()
