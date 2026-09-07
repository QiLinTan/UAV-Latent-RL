"""Continuous future-reference generation from a high-level decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_packet import ReferencePacket


@dataclass(frozen=True)
class HighLevelDecision:
    """The frozen upper-policy output after adapting it to reference space."""

    target_position: np.ndarray
    desired_velocity: np.ndarray
    latent: np.ndarray
    policy_action: np.ndarray | None = None

    def __post_init__(self):
        target_position = np.asarray(self.target_position, dtype=np.float32).reshape(-1)
        desired_velocity = np.asarray(self.desired_velocity, dtype=np.float32).reshape(-1)
        latent = np.asarray(self.latent, dtype=np.float32).reshape(-1)
        if target_position.shape != (3,) or desired_velocity.shape != (3,):
            raise ValueError("target_position and desired_velocity must be three-vectors.")
        if latent.shape != (16,):
            raise ValueError("The prototype requires the frozen 16D task latent.")
        policy_action = None if self.policy_action is None else np.asarray(
            self.policy_action, dtype=np.float32
        ).reshape(-1)
        object.__setattr__(self, "target_position", target_position.copy())
        object.__setattr__(self, "desired_velocity", desired_velocity.copy())
        object.__setattr__(self, "latent", latent.copy())
        object.__setattr__(self, "policy_action", None if policy_action is None else policy_action.copy())


class ReferenceGenerator:
    """Generate acceleration-limited cubic-Hermite reference packets.

    At a switch, the packet starts from the old packet sampled at the exact
    ready time.  If no valid old packet exists, the low-level hold state is the
    continuity anchor.  Both cases make ``p_new(0)`` and ``v_new(0)`` explicit.
    """

    def __init__(
        self,
        *,
        horizon_seconds: float = 0.5,
        sequence_length: int = 16,
        max_speed: float = 0.6,
        max_acceleration: float = 1.2,
        max_vertical_speed: float = 0.35,
    ):
        if horizon_seconds <= 0.0 or sequence_length < 3:
            raise ValueError("horizon_seconds must be positive and sequence_length >= 3.")
        if min(max_speed, max_acceleration, max_vertical_speed) <= 0.0:
            raise ValueError("trajectory limits must be positive.")
        self.horizon_seconds = float(horizon_seconds)
        self.sequence_length = int(sequence_length)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.max_vertical_speed = float(max_vertical_speed)
        self.last_switching_error = 0.0
        self.last_velocity_switching_error = 0.0

    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        value = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
        planar_speed = float(np.linalg.norm(value[:2]))
        if planar_speed > self.max_speed:
            value[:2] *= self.max_speed / planar_speed
        value[2] = np.clip(value[2], -self.max_vertical_speed, self.max_vertical_speed)
        return value

    def _anchor(
        self,
        *,
        previous_packet: ReferencePacket | None,
        t_ready: float,
        fallback_position,
        fallback_velocity,
    ) -> tuple[np.ndarray, np.ndarray]:
        if previous_packet is not None and previous_packet.is_valid(t_ready):
            old = previous_packet.sample(t_ready)
            position, velocity = old.position, old.velocity
        else:
            position = np.asarray(fallback_position, dtype=np.float32).reshape(3)
            velocity = np.asarray(fallback_velocity, dtype=np.float32).reshape(3)
        return position.astype(np.float64), velocity.astype(np.float64)

    def generate(
        self,
        decision: HighLevelDecision,
        *,
        version: int,
        t_generate: float,
        t_ready: float,
        previous_packet: ReferencePacket | None,
        fallback_position,
        fallback_velocity,
    ) -> ReferencePacket:
        """Create a packet whose initial value is continuous with the old one."""

        start_position, start_velocity = self._anchor(
            previous_packet=previous_packet,
            t_ready=t_ready,
            fallback_position=fallback_position,
            fallback_velocity=fallback_velocity,
        )
        requested_velocity = self._clip_velocity(decision.desired_velocity)
        target_delta = np.asarray(decision.target_position, dtype=np.float64) - start_position
        target_speed = self._clip_velocity(target_delta / self.horizon_seconds)
        # Goal direction constrains the endpoint; the policy-derived velocity
        # remains the upper decision if it is already smaller than that bound.
        desired_velocity = self._clip_velocity(
            np.where(np.abs(target_delta) > 1e-6, requested_velocity, target_speed)
        )
        velocity_delta = desired_velocity - start_velocity
        acceleration = np.clip(
            velocity_delta / self.horizon_seconds,
            -self.max_acceleration,
            self.max_acceleration,
        )
        end_velocity = self._clip_velocity(start_velocity + acceleration * self.horizon_seconds)
        end_position = start_position + 0.5 * (start_velocity + end_velocity) * self.horizon_seconds

        times = np.linspace(0.0, self.horizon_seconds, self.sequence_length, dtype=np.float64)
        u = times / self.horizon_seconds
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        positions = (
            h00[:, None] * start_position
            + h10[:, None] * self.horizon_seconds * start_velocity
            + h01[:, None] * end_position
            + h11[:, None] * self.horizon_seconds * end_velocity
        )
        dh00 = (6.0 * u**2 - 6.0 * u) / self.horizon_seconds
        dh10 = 3.0 * u**2 - 4.0 * u + 1.0
        dh01 = (-6.0 * u**2 + 6.0 * u) / self.horizon_seconds
        dh11 = 3.0 * u**2 - 2.0 * u
        velocities = (
            dh00[:, None] * start_position
            + dh10[:, None] * start_velocity
            + dh01[:, None] * end_position
            + dh11[:, None] * end_velocity
        )
        self.last_switching_error = float(np.linalg.norm(positions[0] - start_position))
        self.last_velocity_switching_error = float(np.linalg.norm(velocities[0] - start_velocity))
        return ReferencePacket(
            version=int(version),
            t_generate=float(t_generate),
            t_ready=float(t_ready),
            valid_until=float(t_ready) + self.horizon_seconds,
            positions=positions.astype(np.float32),
            velocities=velocities.astype(np.float32),
            relative_times=times,
            latent=decision.latent,
        )
