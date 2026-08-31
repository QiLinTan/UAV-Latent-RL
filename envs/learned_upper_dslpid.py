from __future__ import annotations

from collections import deque

import numpy as np
import pybullet as p
from gymnasium import Env, spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from envs.observation_layout import FOREST_TASK_DIM, KIN_DIM, ForestObservationLayout
from models.motor_action_codec import ASYMMETRIC_RPM, MotorActionCodec, MotorPhysicalLimits
from models.reference_packet import ReferencePacket


class LearnedUpperDSLPIDEnv(Env):
    """Expose a low-frequency reference action while DSLPID runs at motor rate.

    The wrapped forest environment still receives normalized four-motor RPM
    commands.  One public ``step`` creates one ReferencePacket and executes it
    for ``upper_control_interval`` lower control cycles.  Consequently the
    replay buffer stores the learned upper action, never the hidden motor
    command.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env,
        *,
        upper_control_interval: int = 8,
        reference_sequence_length: int = 15,
        reference_horizon_seconds: float = 1.0,
        max_reference_speed: float = 0.6,
        max_reference_acceleration: float = 1.5,
        max_reference_vertical_speed: float = 0.4,
        max_reference_vertical_acceleration: float = 0.5,
        semantic_history_length: int = 0,
    ):
        super().__init__()
        if int(upper_control_interval) <= 0:
            raise ValueError("upper_control_interval must be positive.")
        if int(base_env.CTRL_FREQ) % int(upper_control_interval) != 0:
            raise ValueError(
                "The lower control frequency must be divisible by upper_control_interval."
            )

        self.base_env = base_env
        self.upper_control_interval = int(upper_control_interval)
        self.lower_ctrl_freq = int(base_env.CTRL_FREQ)
        self.lower_control_dt = float(base_env.CTRL_TIMESTEP)
        self.CTRL_FREQ = self.lower_ctrl_freq // self.upper_control_interval
        self.CTRL_TIMESTEP = self.lower_control_dt * self.upper_control_interval
        self.reference_sequence_length = int(reference_sequence_length)
        self.reference_horizon_seconds = float(reference_horizon_seconds)
        self.max_reference_speed = float(max_reference_speed)
        self.max_reference_acceleration = float(max_reference_acceleration)
        self.max_reference_vertical_speed = float(max_reference_vertical_speed)
        self.max_reference_vertical_acceleration = float(
            max_reference_vertical_acceleration
        )
        self.semantic_history_length = max(0, int(semantic_history_length))
        self.semantic_frame_dim = 8 + 3 + 1
        self._semantic_history = deque(maxlen=max(1, self.semantic_history_length))
        if self.reference_sequence_length < 3:
            raise ValueError("reference_sequence_length must be at least 3.")
        if self.reference_horizon_seconds <= 0.0:
            raise ValueError("reference_horizon_seconds must be positive.")
        if min(
            self.max_reference_speed,
            self.max_reference_acceleration,
            self.max_reference_vertical_speed,
            self.max_reference_vertical_acceleration,
        ) <= 0.0:
            raise ValueError("Velocity and acceleration limits must be positive.")

        base_layout = ForestObservationLayout.from_total_dim(
            int(np.prod(base_env.observation_space.shape)),
            action_dim=int(base_env.action_space.shape[-1]),
        )
        self._base_layout = base_layout
        # The persistent adapter state changes future transitions. Expose it to
        # the upper policy so the 15 Hz process remains observable.
        self.base_upper_obs_dim = KIN_DIM + FOREST_TASK_DIM + 3 + 3
        upper_obs_dim = (
            self.base_upper_obs_dim
            + self.semantic_history_length * self.semantic_frame_dim
        )
        self.observation_space = spaces.Box(
            low=np.full((1, upper_obs_dim), -np.inf, dtype=np.float32),
            high=np.full((1, upper_obs_dim), np.inf, dtype=np.float32),
            shape=(1, upper_obs_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 3),
            dtype=np.float32,
        )
        self.teacher = DSLPIDControl(drone_model=base_env.DRONE_MODEL)
        self.motor_codec = MotorActionCodec(
            MotorPhysicalLimits(
                min_rpm=0.0,
                hover_rpm=float(base_env.HOVER_RPM),
                max_rpm=float(base_env.MAX_RPM),
                kf=float(base_env.KF),
            ),
            mode=ASYMMETRIC_RPM,
        )
        base_env.set_motor_action_codec(self.motor_codec)

        self.runtime_lower_step = 0
        self.last_reference_packet = None
        self.last_reference_sample = None
        self.last_motor_action = np.zeros(4, dtype=np.float32)
        self.reference_position = np.zeros(3, dtype=np.float32)
        self.reference_velocity = np.zeros(3, dtype=np.float32)
        self.command_velocity = np.zeros(3, dtype=np.float32)
        self.reference_acceleration = np.zeros(3, dtype=np.float32)
        self.packet_generation_count = 0

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base_env, name)

    def _upper_observation(self, observation) -> np.ndarray:
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        task = flat[self._base_layout.goal_start : self._base_layout.total_dim]
        actual_position = np.asarray(self.base_env.pos[0], dtype=np.float32)
        reference_error = np.clip(
            (self.reference_position - actual_position) / 0.5,
            -4.0,
            4.0,
        )
        normalized_reference_velocity = np.clip(
            self.reference_velocity / 2.0,
            -1.0,
            1.0,
        )
        current = np.concatenate(
            [
                flat[:KIN_DIM],
                task,
                reference_error,
                normalized_reference_velocity,
            ]
        ).astype(np.float32)
        if self.semantic_history_length:
            history = np.concatenate(tuple(self._semantic_history), axis=0)
            current = np.concatenate([current, history], axis=0)
        return current.reshape(1, -1).astype(np.float32)

    def _semantic_frame(self, observation, tracking_error: float = 0.0) -> np.ndarray:
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        ranges = np.clip(flat[self._base_layout.range_start : self._base_layout.total_dim], 0.0, 1.0)
        command_scale = np.array(
            [self.max_reference_speed, self.max_reference_speed, self.max_reference_vertical_speed],
            dtype=np.float32,
        )
        normalized_command = np.clip(self.command_velocity / command_scale, -1.0, 1.0)
        normalized_tracking_error = np.array(
            [np.clip(float(tracking_error) / 0.5, 0.0, 4.0)], dtype=np.float32
        )
        return np.concatenate([ranges, normalized_command, normalized_tracking_error]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.base_env.reset(seed=seed, options=options)
        self.teacher.reset()
        self.runtime_lower_step = 0
        self.last_reference_packet = None
        self.last_reference_sample = None
        self.last_motor_action.fill(0.0)
        self.reference_position = np.asarray(
            self.base_env.pos[0], dtype=np.float32
        ).copy()
        self.reference_velocity.fill(0.0)
        self.command_velocity.fill(0.0)
        self.reference_acceleration.fill(0.0)
        self.packet_generation_count = 0
        self._semantic_history.clear()
        if self.semantic_history_length:
            initial_frame = self._semantic_frame(observation)
            for _ in range(self.semantic_history_length):
                self._semantic_history.append(initial_frame.copy())
        return self._upper_observation(observation), {
            **dict(info),
            "controller_mode": "learned_upper_dslpid",
            "upper_control_interval": self.upper_control_interval,
            "semantic_history_length": self.semantic_history_length,
        }

    def _velocity_command(self, upper_action) -> np.ndarray:
        action = np.clip(
            np.asarray(upper_action, dtype=np.float32).reshape(3),
            -1.0,
            1.0,
        )
        planar = action[:2].astype(np.float64)
        planar_norm = float(np.linalg.norm(planar))
        if planar_norm > 1.0:
            planar /= planar_norm
        return np.array(
            (
                planar[0] * self.max_reference_speed,
                planar[1] * self.max_reference_speed,
                float(action[2]) * self.max_reference_vertical_speed,
            ),
            dtype=np.float32,
        )

    def _advance_reference_state(self, position, velocity, command, dt):
        position = np.asarray(position, dtype=np.float64).copy()
        velocity = np.asarray(velocity, dtype=np.float64).copy()
        command = np.asarray(command, dtype=np.float64)
        acceleration_limits = np.array(
            [
                self.max_reference_acceleration,
                self.max_reference_acceleration,
                self.max_reference_vertical_acceleration,
            ],
            dtype=np.float64,
        )
        velocity_delta = np.clip(
            command - velocity,
            -acceleration_limits * float(dt),
            acceleration_limits * float(dt),
        )
        next_velocity = velocity + velocity_delta
        planar_speed = float(np.linalg.norm(next_velocity[:2]))
        if planar_speed > self.max_reference_speed:
            next_velocity[:2] *= self.max_reference_speed / planar_speed
        next_velocity[2] = np.clip(
            next_velocity[2],
            -self.max_reference_vertical_speed,
            self.max_reference_vertical_speed,
        )
        acceleration = (next_velocity - velocity) / max(float(dt), 1e-9)
        next_position = position + 0.5 * (velocity + next_velocity) * float(dt)
        return (
            next_position.astype(np.float32),
            next_velocity.astype(np.float32),
            acceleration.astype(np.float32),
        )

    def _generate_velocity_packet(self, command, now: float) -> ReferencePacket:
        timestamps = np.linspace(
            0.0,
            self.reference_horizon_seconds,
            self.reference_sequence_length,
            dtype=np.float64,
        )
        positions = [self.reference_position.copy()]
        velocities = [self.reference_velocity.copy()]
        position = self.reference_position.copy()
        velocity = self.reference_velocity.copy()
        for index in range(1, len(timestamps)):
            dt = float(timestamps[index] - timestamps[index - 1])
            position, velocity, _ = self._advance_reference_state(
                position,
                velocity,
                command,
                dt,
            )
            positions.append(position.copy())
            velocities.append(velocity.copy())
        self.packet_generation_count += 1
        return ReferencePacket(
            positions=np.asarray(positions, dtype=np.float32),
            velocities=np.asarray(velocities, dtype=np.float32),
            relative_timestamps=timestamps,
            t_gen=float(now),
            t_start=float(now),
            t_receive=float(now),
            valid_duration=self.reference_horizon_seconds,
            version=self.packet_generation_count,
            frame_id="world",
            origin_position=np.zeros(3, dtype=np.float32),
            origin_attitude=np.zeros(3, dtype=np.float32),
            code=np.asarray(command, dtype=np.float32),
        )

    def step(self, action):
        command = self._velocity_command(action)
        self.command_velocity = command.copy()
        now = self.runtime_lower_step * self.lower_control_dt
        packet = self._generate_velocity_packet(command, now)
        self.last_reference_packet = packet

        reward_sum = 0.0
        terminated = False
        truncated = False
        info = {}
        observation = None
        tracking_errors = []
        motor_saturation = []

        for _ in range(self.upper_control_interval):
            lower_now = self.runtime_lower_step * self.lower_control_dt
            control_time = lower_now + self.lower_control_dt
            sample = packet.sample(
                control_time,
                lookahead_seconds=0.0,
            )
            self.last_reference_sample = sample
            previous_velocity = self.reference_velocity.copy()
            self.reference_position = np.asarray(
                sample["current_position"], dtype=np.float32
            ).copy()
            self.reference_velocity = np.asarray(
                sample["current_velocity"], dtype=np.float32
            ).copy()
            self.reference_acceleration = (
                self.reference_velocity - previous_velocity
            ) / self.lower_control_dt
            rpm, _, _ = self.teacher.computeControl(
                control_timestep=self.lower_control_dt,
                cur_pos=np.asarray(self.base_env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(
                    p.getQuaternionFromEuler(self.base_env.rpy[0]),
                    dtype=np.float64,
                ),
                cur_vel=np.asarray(self.base_env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(self.base_env.ang_v[0], dtype=np.float64),
                target_pos=self.reference_position,
                target_vel=self.reference_velocity,
            )
            motor_action = self.motor_codec.rpm_to_normalized_action(rpm).astype(np.float32)
            self.last_motor_action = motor_action
            observation, reward, terminated, truncated, info = self.base_env.step(
                motor_action.reshape(1, -1)
            )
            self.runtime_lower_step += 1
            reward_sum += float(reward)
            tracking_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(self.base_env.pos[0])
                        - self.reference_position
                    )
                )
            )
            motor_saturation.append(float(np.mean(np.abs(motor_action) >= 0.999)))
            if terminated or truncated:
                break

        enriched_info = {
            **dict(info),
            "controller_mode": "learned_upper_dslpid",
            "upper_control_interval": self.upper_control_interval,
            "lower_steps_executed": len(tracking_errors),
            "reference_packet_version": int(packet.version),
            "reference_packet_valid": bool(
                self.last_reference_sample is not None
                and self.last_reference_sample["valid"]
            ),
            "reference_endpoint_x": float(packet.to_world(packet.positions[-1])[0]),
            "reference_endpoint_y": float(packet.to_world(packet.positions[-1])[1]),
            "reference_endpoint_z": float(packet.to_world(packet.positions[-1])[2]),
            "velocity_command_x": float(self.command_velocity[0]),
            "velocity_command_y": float(self.command_velocity[1]),
            "velocity_command_z": float(self.command_velocity[2]),
            "reference_velocity_x": float(self.reference_velocity[0]),
            "reference_velocity_y": float(self.reference_velocity[1]),
            "reference_velocity_z": float(self.reference_velocity[2]),
            "actual_velocity_x": float(self.base_env.vel[0][0]),
            "actual_velocity_y": float(self.base_env.vel[0][1]),
            "actual_velocity_z": float(self.base_env.vel[0][2]),
            "reference_position_error": float(
                np.linalg.norm(
                    self.reference_position
                    - np.asarray(self.base_env.pos[0], dtype=np.float32)
                )
            ),
            "reference_acceleration_norm": float(
                np.linalg.norm(self.reference_acceleration)
            ),
            "trajectory_age_seconds": float(
                self.last_reference_sample["execution_time"]
            ),
            "packet_generation_count": self.packet_generation_count,
            "reference_tracking_error_mean": (
                float(np.mean(tracking_errors)) if tracking_errors else float("nan")
            ),
            "motor_action_saturation_mean": (
                float(np.mean(motor_saturation)) if motor_saturation else float("nan")
            ),
        }
        if self.semantic_history_length:
            self._semantic_history.append(
                self._semantic_frame(
                    observation,
                    float(np.mean(tracking_errors)) if tracking_errors else 0.0,
                )
            )
        return (
            self._upper_observation(observation),
            reward_sum,
            bool(terminated),
            bool(truncated),
            enriched_info,
        )

    def close(self):
        return self.base_env.close()

    def record_episode_outcome(self, success: bool):
        if hasattr(self.base_env, "record_episode_outcome"):
            return self.base_env.record_episode_outcome(success)
        return None


class LearnedUpperDSLPIDForestEnv(LearnedUpperDSLPIDEnv):
    """Constructible forest specialization used by training/evaluation callbacks."""

    _WRAPPER_KEYS = {
        "upper_control_interval",
        "reference_sequence_length",
        "reference_horizon_seconds",
        "max_reference_speed",
        "max_reference_acceleration",
        "max_reference_vertical_speed",
        "max_reference_vertical_acceleration",
        "semantic_history_length",
    }

    def __init__(self, **kwargs):
        from envs.ForestAviary import CustomForestAviary

        wrapper_kwargs = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in self._WRAPPER_KEYS
        }
        super().__init__(CustomForestAviary(**kwargs), **wrapper_kwargs)
