"""The immutable high-level to low-level reference contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceSample:
    """A time-interpolated command consumed by one low-level control cycle."""

    position: np.ndarray
    velocity: np.ndarray
    packet_age: float
    version: int


@dataclass(frozen=True)
class ReferencePacket:
    """An atomic, time-stamped future reference sequence.

    ``positions`` and ``velocities`` are world-frame values.  A packet only
    becomes executable at ``t_ready`` and is stale after ``valid_until``.  The
    optional latent is deliberately carried even though the first DSLPID lower
    controller does not consume it.
    """

    version: int
    t_generate: float
    t_ready: float
    valid_until: float
    positions: np.ndarray
    velocities: np.ndarray
    relative_times: np.ndarray
    latent: np.ndarray | None = None

    def __post_init__(self):
        positions = np.asarray(self.positions, dtype=np.float32)
        velocities = np.asarray(self.velocities, dtype=np.float32)
        relative_times = np.asarray(self.relative_times, dtype=np.float64).reshape(-1)
        latent = None if self.latent is None else np.asarray(self.latent, dtype=np.float32).reshape(-1)

        if int(self.version) < 1:
            raise ValueError("version must be a positive integer.")
        if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [N, 3] with N >= 2.")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same shape as positions.")
        if relative_times.shape != (positions.shape[0],):
            raise ValueError("relative_times must contain one timestamp per point.")
        if not np.isclose(relative_times[0], 0.0):
            raise ValueError("relative_times must begin at 0.0.")
        if np.any(np.diff(relative_times) <= 0.0):
            raise ValueError("relative_times must be strictly increasing.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("reference positions and velocities must be finite.")
        if not np.all(np.isfinite(relative_times)):
            raise ValueError("relative_times must be finite.")
        if not np.isfinite(float(self.t_generate)) or not np.isfinite(float(self.t_ready)):
            raise ValueError("timestamps must be finite.")
        if float(self.t_ready) < float(self.t_generate):
            raise ValueError("t_ready cannot precede t_generate.")
        if float(self.valid_until) <= float(self.t_ready):
            raise ValueError("valid_until must be later than t_ready.")
        if latent is not None and latent.shape != (16,):
            raise ValueError("latent must be a 16-dimensional vector when present.")

        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "velocities", velocities.copy())
        object.__setattr__(self, "relative_times", relative_times.copy())
        object.__setattr__(self, "latent", None if latent is None else latent.copy())

    @property
    def horizon(self) -> float:
        return float(self.relative_times[-1])

    def is_valid(self, current_time: float) -> bool:
        now = float(current_time)
        return bool(float(self.t_ready) <= now <= float(self.valid_until))

    def packet_age(self, current_time: float) -> float:
        return max(0.0, float(current_time) - float(self.t_generate))

    def sample(self, current_time: float) -> ReferenceSample:
        """Interpolate a valid packet at ``current_time``.

        Callers must select validity through :class:`AsyncBuffer` first.  This
        method clamps the final segment so delayed lower cycles never generate a
        reference discontinuity from numerical boundary noise.
        """

        elapsed = float(np.clip(
            float(current_time) - float(self.t_ready),
            self.relative_times[0],
            self.relative_times[-1],
        ))
        upper = int(np.searchsorted(self.relative_times, elapsed, side="right"))
        upper = min(max(upper, 1), len(self.relative_times) - 1)
        lower = upper - 1
        dt = max(float(self.relative_times[upper] - self.relative_times[lower]), 1e-12)
        alpha = float(np.clip(
            (elapsed - float(self.relative_times[lower])) / dt,
            0.0,
            1.0,
        ))
        position = (1.0 - alpha) * self.positions[lower] + alpha * self.positions[upper]
        velocity = (1.0 - alpha) * self.velocities[lower] + alpha * self.velocities[upper]
        return ReferenceSample(
            position=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            packet_age=self.packet_age(current_time),
            version=int(self.version),
        )
