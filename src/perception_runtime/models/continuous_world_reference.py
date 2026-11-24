"""Deterministic world-frame trajectories and sliding ReferencePacket windows.

This module is shared by the Stage 1 DSLPID gate and Stage 2 learned lower
controller.  It deliberately has no RL, planner, environment, or controller
dependency.
"""

from __future__ import annotations

import numpy as np

from models.reference_packet import ReferencePacket


class TrapezoidalWorldTrajectory:
    """C1 position/velocity trajectory with bounded scalar acceleration."""

    def __init__(self, start_position, target_position, *, max_speed: float, max_acceleration: float):
        self.start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
        self.target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        delta = self.target_position - self.start_position
        self.distance = float(np.linalg.norm(delta))
        if self.distance <= 0.0:
            raise ValueError("Start and target positions must be different.")
        self.direction = delta / self.distance
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        if min(self.max_speed, self.max_acceleration) <= 0.0:
            raise ValueError("max_speed and max_acceleration must be positive.")

        nominal_accel_time = self.max_speed / self.max_acceleration
        nominal_accel_distance = 0.5 * self.max_acceleration * nominal_accel_time**2
        if 2.0 * nominal_accel_distance <= self.distance:
            self.peak_speed = self.max_speed
            self.accel_time = nominal_accel_time
            self.accel_distance = nominal_accel_distance
            self.cruise_time = (
                self.distance - 2.0 * self.accel_distance
            ) / self.peak_speed
        else:
            self.peak_speed = float(np.sqrt(self.distance * self.max_acceleration))
            self.accel_time = self.peak_speed / self.max_acceleration
            self.accel_distance = 0.5 * self.max_acceleration * self.accel_time**2
            self.cruise_time = 0.0
        self.duration = 2.0 * self.accel_time + self.cruise_time

    def sample(self, time_seconds) -> tuple[np.ndarray, np.ndarray]:
        """Return world position and velocity at scalar or array simulation time."""

        times = np.asarray(time_seconds, dtype=np.float64)
        clipped = np.clip(times, 0.0, self.duration)
        scalar_distance = np.empty_like(clipped)
        scalar_speed = np.empty_like(clipped)

        accelerating = clipped < self.accel_time
        cruising = np.logical_and(
            clipped >= self.accel_time,
            clipped < self.accel_time + self.cruise_time,
        )
        decelerating = np.logical_not(np.logical_or(accelerating, cruising))

        scalar_distance[accelerating] = (
            0.5 * self.max_acceleration * clipped[accelerating] ** 2
        )
        scalar_speed[accelerating] = self.max_acceleration * clipped[accelerating]

        cruise_elapsed = clipped[cruising] - self.accel_time
        scalar_distance[cruising] = (
            self.accel_distance + self.peak_speed * cruise_elapsed
        )
        scalar_speed[cruising] = self.peak_speed

        remaining = self.duration - clipped[decelerating]
        scalar_distance[decelerating] = (
            self.distance - 0.5 * self.max_acceleration * remaining**2
        )
        scalar_speed[decelerating] = self.max_acceleration * remaining

        before_start = times <= 0.0
        after_end = times >= self.duration
        scalar_distance = np.where(before_start, 0.0, scalar_distance)
        scalar_speed = np.where(before_start, 0.0, scalar_speed)
        scalar_distance = np.where(after_end, self.distance, scalar_distance)
        scalar_speed = np.where(after_end, 0.0, scalar_speed)

        positions = self.start_position + np.expand_dims(scalar_distance, -1) * self.direction
        velocities = np.expand_dims(scalar_speed, -1) * self.direction
        return positions.astype(np.float32), velocities.astype(np.float32)


class ConstantWorldTrajectory:
    """Fixed world position with zero velocity, used by the Stage 2 hover gate."""

    def __init__(self, position, *, duration: float):
        self.start_position = np.asarray(position, dtype=np.float64).reshape(3)
        self.target_position = self.start_position.copy()
        self.distance = 0.0
        self.duration = float(duration)
        self.max_speed = 0.0
        self.max_acceleration = 0.0
        if self.duration <= 0.0:
            raise ValueError("Hover duration must be positive.")

    def sample(self, time_seconds) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(time_seconds, dtype=np.float64)
        positions = np.broadcast_to(
            self.start_position,
            times.shape + (3,),
        ).copy()
        velocities = np.zeros(times.shape + (3,), dtype=np.float64)
        return positions.astype(np.float32), velocities.astype(np.float32)


class ContinuousWorldPacketSource:
    """Publish sliding windows sampled from one immutable global trajectory."""

    def __init__(
        self,
        trajectory,
        *,
        window_seconds: float = 1.0,
        sequence_length: int = 16,
    ):
        if float(window_seconds) <= 0.0 or int(sequence_length) < 2:
            raise ValueError("A positive window and at least two samples are required.")
        if not hasattr(trajectory, "sample"):
            raise TypeError("trajectory must expose sample(time_seconds).")
        self.trajectory = trajectory
        self.window_seconds = float(window_seconds)
        self.sequence_length = int(sequence_length)
        self.relative_timestamps = np.linspace(
            0.0,
            self.window_seconds,
            self.sequence_length,
            dtype=np.float64,
        )
        self.version = 0

    def make_packet(self, simulation_time: float) -> ReferencePacket:
        now = float(simulation_time)
        positions, velocities = self.trajectory.sample(now + self.relative_timestamps)
        self.version += 1
        return ReferencePacket(
            positions=positions,
            velocities=velocities,
            relative_timestamps=self.relative_timestamps,
            t_gen=now,
            t_start=now,
            t_receive=now,
            valid_duration=self.window_seconds,
            version=self.version,
            frame_id="world",
            origin_position=np.zeros(3, dtype=np.float32),
            origin_attitude=np.zeros(3, dtype=np.float32),
            code=np.array(
                [
                    float(getattr(self.trajectory, "max_speed", 0.0)),
                    float(getattr(self.trajectory, "max_acceleration", 0.0)),
                ],
                dtype=np.float32,
            ),
        )


def derive_motion_timing(
    trajectory: TrapezoidalWorldTrajectory,
    *,
    max_speed: float,
    settle_seconds: float,
) -> dict[str, float | bool]:
    """Derive a reachable timeout instead of relying on a fixed episode length."""

    physical_lower_bound = trajectory.distance / float(max_speed)
    timeout = trajectory.duration + float(settle_seconds)
    reachable = bool(
        trajectory.duration + 1e-12 >= physical_lower_bound
        and timeout + 1e-12 >= trajectory.duration + float(settle_seconds)
    )
    if not reachable:
        raise AssertionError(
            "Reference timeout is shorter than the speed-limited reachable trajectory."
        )
    return {
        "physical_reachability_lower_bound": float(physical_lower_bound),
        "trajectory_duration": float(trajectory.duration),
        "timeout": float(timeout),
        "reachability_assertion": True,
    }
