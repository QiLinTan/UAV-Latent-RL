from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ReferencePacket:
    """Atomic high-to-low reference message with explicit space-time semantics.

    Positions and velocities are expressed in ``frame_id``.  The first
    implementation uses an axis-aligned local navigation frame whose world
    origin is ``origin_position``.
    """

    positions: np.ndarray
    velocities: np.ndarray
    relative_timestamps: np.ndarray
    t_gen: float
    t_start: float
    t_receive: float
    valid_duration: float
    version: int
    frame_id: str
    origin_position: np.ndarray
    origin_attitude: np.ndarray
    valid: bool = True
    code: np.ndarray | None = None

    def __post_init__(self):
        positions = np.asarray(self.positions, dtype=np.float32)
        velocities = np.asarray(self.velocities, dtype=np.float32)
        timestamps = np.asarray(self.relative_timestamps, dtype=np.float64).reshape(-1)
        origin_position = np.asarray(self.origin_position, dtype=np.float32).reshape(-1)
        origin_attitude = np.asarray(self.origin_attitude, dtype=np.float32).reshape(-1)
        code = None if self.code is None else np.asarray(self.code, dtype=np.float32).reshape(-1)

        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("positions must have shape [N, 3] with N >= 2.")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same shape as positions.")
        if timestamps.shape != (positions.shape[0],):
            raise ValueError("relative_timestamps must contain one value per reference point.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Reference positions and velocities must be finite.")
        if not np.all(np.isfinite(timestamps)) or timestamps[0] < 0.0:
            raise ValueError("Reference timestamps must be finite and non-negative.")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Reference timestamps must be strictly increasing.")
        if float(self.valid_duration) <= 0.0:
            raise ValueError("valid_duration must be positive.")
        if int(self.version) < 1:
            raise ValueError("version must be positive.")
        if not str(self.frame_id):
            raise ValueError("frame_id must be non-empty.")
        if origin_position.shape != (3,) or origin_attitude.shape != (3,):
            raise ValueError("origin_position and origin_attitude must be three-vectors.")
        if float(self.t_start) < float(self.t_gen):
            raise ValueError("t_start cannot be earlier than t_gen.")
        if float(self.t_receive) < float(self.t_gen):
            raise ValueError("t_receive cannot be earlier than t_gen.")

        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "velocities", velocities.copy())
        object.__setattr__(self, "relative_timestamps", timestamps.copy())
        object.__setattr__(self, "origin_position", origin_position.copy())
        object.__setattr__(self, "origin_attitude", origin_attitude.copy())
        object.__setattr__(self, "code", None if code is None else code.copy())

    @property
    def expires_at(self) -> float:
        return float(self.t_start + self.valid_duration)

    def with_receive_time(self, t_receive: float) -> "ReferencePacket":
        return replace(self, t_receive=float(t_receive))

    def to_world(self, local_value: np.ndarray, *, is_velocity: bool = False) -> np.ndarray:
        """Transform a local vector to world coordinates.

        The initial implementation intentionally supports an axis-aligned local
        frame only.  Keeping the frame identifier explicit prevents silently
        treating body-frame and world-frame references as interchangeable.
        """

        if self.frame_id not in {"local_navigation", "world"}:
            raise ValueError(f"Unsupported reference frame: {self.frame_id!r}.")
        value = np.asarray(local_value, dtype=np.float32)
        if self.frame_id == "world" or is_velocity:
            return value.copy()
        return value + self.origin_position

    def sample(self, now: float, lookahead_seconds: float = 0.0) -> dict:
        execution_time = max(0.0, float(now) - float(self.t_start))
        query_time = execution_time
        current_position, current_velocity, lower_index = self._interpolate(query_time)
        lookahead_position, lookahead_velocity, lookahead_index = self._interpolate(
            query_time + max(0.0, float(lookahead_seconds))
        )
        age = max(0.0, float(now) - float(self.t_gen))
        valid = bool(
            self.valid
            and float(now) >= float(self.t_start)
            and float(now) <= self.expires_at
        )
        return {
            "current_position": self.to_world(current_position),
            "current_velocity": self.to_world(current_velocity, is_velocity=True),
            "lookahead_position": self.to_world(lookahead_position),
            "lookahead_velocity": self.to_world(lookahead_velocity, is_velocity=True),
            "age_seconds": age,
            "age_ratio": float(np.clip(age / self.valid_duration, 0.0, 1.0)),
            "valid": valid,
            "version": int(self.version),
            "current_index": int(lower_index),
            "lookahead_index": int(lookahead_index),
            "execution_time": execution_time,
            "frame_id": self.frame_id,
        }

    def _interpolate(self, relative_time: float):
        timestamps = self.relative_timestamps
        t = float(np.clip(relative_time, timestamps[0], timestamps[-1]))
        upper = int(np.searchsorted(timestamps, t, side="right"))
        upper = min(max(upper, 1), len(timestamps) - 1)
        lower = upper - 1
        interval = max(float(timestamps[upper] - timestamps[lower]), 1e-9)
        alpha = float(np.clip((t - timestamps[lower]) / interval, 0.0, 1.0))
        position = (1.0 - alpha) * self.positions[lower] + alpha * self.positions[upper]
        velocity = (1.0 - alpha) * self.velocities[lower] + alpha * self.velocities[upper]
        return position.astype(np.float32), velocity.astype(np.float32), lower


class AsyncReferenceBuffer:
    """Atomically stores the newest valid reference packet."""

    def __init__(self):
        self._packet: ReferencePacket | None = None
        self.accepted_packets = 0
        self.rejected_packets = 0
        self.rejected_stale_version = 0
        self.rejected_out_of_order = 0

    @property
    def packet(self) -> ReferencePacket | None:
        return self._packet

    @property
    def has_value(self) -> bool:
        return self._packet is not None

    @property
    def version(self) -> int:
        return 0 if self._packet is None else int(self._packet.version)

    def clear(self):
        self._packet = None

    def publish(self, packet: ReferencePacket) -> bool:
        if not isinstance(packet, ReferencePacket):
            raise TypeError("AsyncReferenceBuffer accepts ReferencePacket values only.")
        if self._packet is not None:
            stale_version = packet.version <= self._packet.version
            out_of_order = packet.t_receive < self._packet.t_receive
            if stale_version or out_of_order:
                self.rejected_packets += 1
                self.rejected_stale_version += int(stale_version)
                self.rejected_out_of_order += int(out_of_order)
                return False
        self._packet = packet
        self.accepted_packets += 1
        return True

    def sample(self, now: float, lookahead_seconds: float = 0.0) -> dict:
        if self._packet is None:
            raise RuntimeError("The asynchronous reference buffer is empty.")
        return self._packet.sample(now, lookahead_seconds=lookahead_seconds)


@dataclass(frozen=True)
class TrajectoryLimits:
    max_speed: float = 0.8
    max_acceleration: float = 2.0
    max_vertical_speed: float = 0.5


class TrajectoryFeasibilityChecker:
    """Kinematic feasibility checks; this is not a complete safety proof."""

    def __init__(self, limits: TrajectoryLimits):
        self.limits = limits

    def check(self, packet: ReferencePacket) -> tuple[bool, dict]:
        timestamps = packet.relative_timestamps
        dt = np.diff(timestamps)
        segment_velocity = np.diff(packet.positions, axis=0) / dt[:, None]
        if len(segment_velocity) > 1:
            accel_dt = 0.5 * (dt[1:] + dt[:-1])
            acceleration = np.diff(segment_velocity, axis=0) / accel_dt[:, None]
            max_acceleration = float(np.linalg.norm(acceleration, axis=1).max())
        else:
            max_acceleration = 0.0
        max_speed = float(np.linalg.norm(segment_velocity, axis=1).max())
        max_vertical_speed = float(np.abs(segment_velocity[:, 2]).max())
        metrics = {
            "max_speed": max_speed,
            "max_acceleration": max_acceleration,
            "max_vertical_speed": max_vertical_speed,
        }
        feasible = (
            max_speed <= self.limits.max_speed + 1e-5
            and max_acceleration <= self.limits.max_acceleration + 1e-5
            and max_vertical_speed <= self.limits.max_vertical_speed + 1e-5
        )
        return bool(feasible), metrics


class RuleReferenceGenerator:
    """Generate conservative minimum-jerk local references for lower-level tests."""

    def __init__(
        self,
        *,
        sequence_length: int = 15,
        horizon_seconds: float = 1.0,
        limits: TrajectoryLimits | None = None,
        mode: str = "line",
    ):
        if sequence_length < 3:
            raise ValueError("sequence_length must be at least 3.")
        if horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive.")
        if mode not in {"line", "hover"}:
            raise ValueError("mode must be 'line' or 'hover'.")
        self.sequence_length = int(sequence_length)
        self.horizon_seconds = float(horizon_seconds)
        self.limits = limits or TrajectoryLimits()
        self.mode = mode
        self._version = 0
        self.checker = TrajectoryFeasibilityChecker(self.limits)

    def generate(
        self,
        *,
        position,
        velocity,
        target_position,
        t_gen: float,
        t_start: float | None = None,
        t_receive: float | None = None,
    ) -> ReferencePacket:
        position = np.asarray(position, dtype=np.float32).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float32).reshape(3)
        target = np.asarray(target_position, dtype=np.float32).reshape(3)
        horizon = self.horizon_seconds
        timestamps = np.linspace(0.0, horizon, self.sequence_length, dtype=np.float64)

        delta = target - position
        if self.mode != "hover":
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                max_displacement_speed = self.limits.max_speed * horizon / 1.875
                max_displacement_accel = self.limits.max_acceleration * horizon**2 / 5.8
                max_displacement = min(max_displacement_speed, max_displacement_accel)
                delta *= min(1.0, max_displacement / distance)
            vertical_limit = self.limits.max_vertical_speed * horizon / 1.875
            delta[2] = float(np.clip(delta[2], -vertical_limit, vertical_limit))
        else:
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                max_displacement = min(
                    self.limits.max_speed * horizon / 1.875,
                    self.limits.max_acceleration * horizon**2 / 5.8,
                )
                delta *= min(1.0, max_displacement / distance)

        u = (timestamps / horizon).astype(np.float32)
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_rate = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / horizon
        positions = blend[:, None] * delta[None, :]
        velocities = blend_rate[:, None] * delta[None, :]

        self._version += 1
        start = float(t_gen if t_start is None else t_start)
        receive = float(t_gen if t_receive is None else t_receive)
        packet = ReferencePacket(
            positions=positions,
            velocities=velocities,
            relative_timestamps=timestamps,
            t_gen=float(t_gen),
            t_start=start,
            t_receive=receive,
            valid_duration=horizon,
            version=self._version,
            frame_id="local_navigation",
            origin_position=position,
            origin_attitude=np.zeros(3, dtype=np.float32),
        )
        feasible, metrics = self.checker.check(packet)
        if not feasible:
            raise ValueError(f"Generated trajectory violates configured limits: {metrics}.")
        return packet


class ActuatorConstraintLayer:
    """Preserve collective/differential structure while enforcing action bounds."""

    def __init__(
        self,
        action_dim: int,
        max_action: float,
        collective_fraction: float,
        differential_fraction: float,
        max_delta: float | None = None,
    ):
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.collective_limit = self.max_action * float(collective_fraction)
        self.differential_limit = self.max_action * float(differential_fraction)
        self.max_delta = None if max_delta is None else float(max_delta)

    def __call__(self, action, previous_action=None) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        collective = float(np.clip(action.mean(), -self.collective_limit, self.collective_limit))
        differential = action - float(action.mean())
        differential -= float(differential.mean())
        max_abs = float(np.abs(differential).max())
        if max_abs > self.differential_limit:
            differential *= self.differential_limit / max(max_abs, 1e-6)
        constrained = np.clip(
            collective + differential,
            -self.max_action,
            self.max_action,
        ).astype(np.float32)
        if previous_action is not None and self.max_delta is not None:
            previous = np.asarray(previous_action, dtype=np.float32).reshape(self.action_dim)
            constrained = np.clip(
                constrained,
                previous - self.max_delta,
                previous + self.max_delta,
            )
        return constrained.astype(np.float32)


class DegradedHoverController:
    """Rate-decay toward hover-equivalent RPM command after reference failure.

    This is a deterministic degradation behavior, not a certified safety
    controller.  A geometric/PID recovery controller can replace it later.
    """

    def __init__(self, action_dim: int, decay: float = 0.15):
        self.action_dim = int(action_dim)
        self.decay = float(np.clip(decay, 0.0, 1.0))

    def __call__(self, previous_action) -> np.ndarray:
        previous = np.asarray(previous_action, dtype=np.float32).reshape(self.action_dim)
        return ((1.0 - self.decay) * previous).astype(np.float32)
