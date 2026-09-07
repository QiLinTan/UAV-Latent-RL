"""Feature extraction and safe conversion of planner output to ReferencePacket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from async_control.reference_packet import ReferencePacket
from envs.observation_layout import ForestObservationLayout


LATENT_DIM = 16
GOAL_FEATURE_DIM = 3
OBSTACLE_FEATURE_DIM = 8
PLANNER_INPUT_DIM = LATENT_DIM + GOAL_FEATURE_DIM + OBSTACLE_FEATURE_DIM
FUTURE_TIMES = np.arange(1, 6, dtype=np.float64) * 0.1


def extract_planner_input(
    latent,
    preprocessed_observation,
    *,
    action_dim: int = 4,
) -> np.ndarray:
    """Extract ``latent + relative goal + range distances`` as exactly 27D."""

    z = np.asarray(latent, dtype=np.float32).reshape(-1)
    observation = np.asarray(preprocessed_observation, dtype=np.float32).reshape(-1)
    if z.shape != (LATENT_DIM,):
        raise ValueError(f"Expected a {LATENT_DIM}D latent, got {z.shape}.")
    layout = ForestObservationLayout.from_total_dim(
        observation.size,
        action_dim=int(action_dim),
    )
    parts = layout.split(observation)
    planner_input = np.concatenate([z, parts.goal, parts.ranges]).astype(np.float32)
    if planner_input.shape != (PLANNER_INPUT_DIM,):
        raise AssertionError(f"Planner input contract produced {planner_input.shape}.")
    return planner_input


@dataclass(frozen=True)
class ReferenceSequenceDecision:
    """Five future relative positions/velocities produced at high-level start."""

    position_offsets: np.ndarray
    velocities: np.ndarray
    relative_times: np.ndarray
    current_velocity: np.ndarray
    latent: np.ndarray

    def __post_init__(self):
        positions = np.asarray(self.position_offsets, dtype=np.float32)
        velocities = np.asarray(self.velocities, dtype=np.float32)
        times = np.asarray(self.relative_times, dtype=np.float64).reshape(-1)
        current_velocity = np.asarray(self.current_velocity, dtype=np.float32).reshape(-1)
        latent = np.asarray(self.latent, dtype=np.float32).reshape(-1)
        if positions.shape != (5, 3) or velocities.shape != (5, 3):
            raise ValueError("Planner decision positions and velocities must have shape [5, 3].")
        if times.shape != (5,) or not np.allclose(times, FUTURE_TIMES):
            raise ValueError("Planner decision timestamps must be [0.1, ..., 0.5] seconds.")
        if current_velocity.shape != (3,) or latent.shape != (LATENT_DIM,):
            raise ValueError("current_velocity must be 3D and latent must be 16D.")
        if not all(np.all(np.isfinite(x)) for x in (positions, velocities, current_velocity, latent)):
            raise ValueError("Planner decision values must be finite.")
        object.__setattr__(self, "position_offsets", positions.copy())
        object.__setattr__(self, "velocities", velocities.copy())
        object.__setattr__(self, "relative_times", times.copy())
        object.__setattr__(self, "current_velocity", current_velocity.copy())
        object.__setattr__(self, "latent", latent.copy())


def _clip_planar_and_vertical(velocity, max_speed, max_vertical_speed):
    value = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    planar = float(np.linalg.norm(value[:2]))
    if planar > float(max_speed):
        value[:2] *= float(max_speed) / planar
    value[2] = np.clip(value[2], -float(max_vertical_speed), float(max_vertical_speed))
    return value


def _interpolate_or_extrapolate(times, values, query, terminal_velocity=None):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = float(query)
    if query <= times[0]:
        if times[0] <= 1e-12:
            return values[0].copy()
        alpha = max(0.0, query / times[0])
        return alpha * values[0]
    if query >= times[-1]:
        if terminal_velocity is None:
            return values[-1].copy()
        return values[-1] + (query - times[-1]) * np.asarray(terminal_velocity)
    upper = int(np.searchsorted(times, query, side="right"))
    lower = upper - 1
    alpha = (query - times[lower]) / (times[upper] - times[lower])
    return (1.0 - alpha) * values[lower] + alpha * values[upper]


class ReferenceSequencePacketGenerator:
    """Delay-compensate, constrain and atomically package a learned sequence."""

    def __init__(
        self,
        *,
        max_speed: float = 0.8,
        max_acceleration: float = 2.0,
        max_vertical_speed: float = 0.4,
    ):
        if min(max_speed, max_acceleration, max_vertical_speed) <= 0.0:
            raise ValueError("Reference sequence limits must be positive.")
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.max_vertical_speed = float(max_vertical_speed)
        self.last_switching_error = 0.0
        self.last_velocity_switching_error = 0.0

    @staticmethod
    def _anchor(previous_packet, t_ready, fallback_position, fallback_velocity):
        if previous_packet is not None and previous_packet.is_valid(t_ready):
            old = previous_packet.sample(t_ready)
            return old.position.astype(np.float64), old.velocity.astype(np.float64)
        return (
            np.asarray(fallback_position, dtype=np.float64).reshape(3),
            np.asarray(fallback_velocity, dtype=np.float64).reshape(3),
        )

    def generate(
        self,
        decision: ReferenceSequenceDecision,
        *,
        version: int,
        t_generate: float,
        t_ready: float,
        previous_packet: ReferencePacket | None,
        fallback_position,
        fallback_velocity,
    ) -> ReferencePacket:
        anchor_position, anchor_velocity = self._anchor(
            previous_packet,
            t_ready,
            fallback_position,
            fallback_velocity,
        )
        age = max(0.0, float(t_ready) - float(t_generate))
        canonical_times = np.concatenate([[0.0], decision.relative_times])
        canonical_offsets = np.vstack(
            [np.zeros(3, dtype=np.float64), decision.position_offsets]
        )
        canonical_velocities = np.vstack(
            [decision.current_velocity, decision.velocities]
        ).astype(np.float64)
        offset_at_ready = _interpolate_or_extrapolate(
            canonical_times,
            canonical_offsets,
            age,
            terminal_velocity=canonical_velocities[-1],
        )

        desired_positions = []
        desired_velocities = []
        for relative_time in FUTURE_TIMES:
            query = age + float(relative_time)
            offset = _interpolate_or_extrapolate(
                canonical_times,
                canonical_offsets,
                query,
                terminal_velocity=canonical_velocities[-1],
            )
            velocity = _interpolate_or_extrapolate(
                canonical_times,
                canonical_velocities,
                query,
            )
            desired_positions.append(anchor_position + offset - offset_at_ready)
            desired_velocities.append(velocity)

        # Convert raw predictions into a dynamically conservative sequence.  A
        # velocity command combines the learned velocity with the direction to
        # the learned position, then obeys speed and acceleration limits.
        positions = [anchor_position.copy()]
        velocities = [anchor_velocity.copy()]
        previous_position = anchor_position.copy()
        previous_velocity = _clip_planar_and_vertical(
            anchor_velocity,
            self.max_speed,
            self.max_vertical_speed,
        )
        previous_time = 0.0
        for point_time, predicted_position, predicted_velocity in zip(
            FUTURE_TIMES,
            desired_positions,
            desired_velocities,
        ):
            dt = float(point_time - previous_time)
            position_velocity = (
                np.asarray(predicted_position, dtype=np.float64) - previous_position
            ) / max(dt, 1e-9)
            command_velocity = _clip_planar_and_vertical(
                0.5 * np.asarray(predicted_velocity) + 0.5 * position_velocity,
                self.max_speed,
                self.max_vertical_speed,
            )
            delta_velocity = command_velocity - previous_velocity
            delta_norm = float(np.linalg.norm(delta_velocity))
            max_delta = self.max_acceleration * dt
            if delta_norm > max_delta:
                delta_velocity *= max_delta / delta_norm
            next_velocity = _clip_planar_and_vertical(
                previous_velocity + delta_velocity,
                self.max_speed,
                self.max_vertical_speed,
            )
            next_position = previous_position + 0.5 * (
                previous_velocity + next_velocity
            ) * dt
            positions.append(next_position.copy())
            velocities.append(next_velocity.copy())
            previous_position = next_position
            previous_velocity = next_velocity
            previous_time = float(point_time)

        packet_times = np.concatenate([[0.0], FUTURE_TIMES])
        positions_value = np.asarray(positions, dtype=np.float32)
        velocities_value = np.asarray(velocities, dtype=np.float32)
        self.last_switching_error = float(
            np.linalg.norm(positions_value[0] - anchor_position)
        )
        self.last_velocity_switching_error = float(
            np.linalg.norm(velocities_value[0] - anchor_velocity)
        )
        return ReferencePacket(
            version=int(version),
            t_generate=float(t_generate),
            t_ready=float(t_ready),
            valid_until=float(t_ready) + float(FUTURE_TIMES[-1]),
            positions=positions_value,
            velocities=velocities_value,
            relative_times=packet_times,
            latent=decision.latent,
        )
