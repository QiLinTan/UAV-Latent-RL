"""Deterministic logical-asynchronous scheduling and lower-loop fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .async_buffer import AsyncBuffer
from .reference_generator import HighLevelDecision, ReferenceGenerator
from .reference_packet import ReferencePacket, ReferenceSample


@dataclass(frozen=True)
class AsyncSchedulerConfig:
    high_frequency_hz: float = 15.0
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 0.0
    skip_probability: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if float(self.high_frequency_hz) <= 0.0:
            raise ValueError("high_frequency_hz must be positive.")
        if float(self.delay_min_seconds) < 0.0:
            raise ValueError("delay_min_seconds cannot be negative.")
        if float(self.delay_max_seconds) < float(self.delay_min_seconds):
            raise ValueError("delay_max_seconds must be >= delay_min_seconds.")
        if not 0.0 <= float(self.skip_probability) <= 1.0:
            raise ValueError("skip_probability must be in [0, 1].")


@dataclass(frozen=True)
class _PendingUpperResult:
    version: int
    t_generate: float
    ready_at: float
    decision: HighLevelDecision


class AsyncScheduler:
    """Model upper computation completion without threads or lower-loop waits."""

    def __init__(
        self,
        config: AsyncSchedulerConfig,
        generator: ReferenceGenerator,
        buffer: AsyncBuffer,
    ):
        self.config = config
        self.generator = generator
        self.buffer = buffer
        self._rng = np.random.default_rng(int(config.seed))
        self._period = 1.0 / float(config.high_frequency_hz)
        self._next_start = 0.0
        self._next_version = 1
        self._pending: list[_PendingUpperResult] = []
        self.high_level_starts = 0
        self.high_level_skips = 0
        self.high_level_completed = 0
        self.commit_rejections = 0
        self.maximum_pending = 0
        self.delays: list[float] = []
        self.switching_errors: list[float] = []
        self.velocity_switching_errors: list[float] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _complete_ready(
        self,
        now: float,
        *,
        fallback_position,
        fallback_velocity,
    ) -> None:
        ready = [job for job in self._pending if job.ready_at <= now + 1e-12]
        self._pending = [job for job in self._pending if job.ready_at > now + 1e-12]
        for job in sorted(ready, key=lambda item: (item.ready_at, item.version)):
            packet = self.generator.generate(
                job.decision,
                version=job.version,
                t_generate=job.t_generate,
                t_ready=now,
                previous_packet=self.buffer.latest_packet,
                fallback_position=fallback_position,
                fallback_velocity=fallback_velocity,
            )
            accepted = self.buffer.commit(packet)
            self.high_level_completed += 1
            self.commit_rejections += int(not accepted)
            if accepted:
                self.delays.append(packet.t_ready - packet.t_generate)
                self.switching_errors.append(self.generator.last_switching_error)
                self.velocity_switching_errors.append(
                    self.generator.last_velocity_switching_error
                )

    def advance(
        self,
        current_time: float,
        decision_factory: Callable[[float], HighLevelDecision],
        *,
        fallback_position,
        fallback_velocity,
    ) -> None:
        """Advance logical time; this function never blocks for an upper result."""

        now = float(current_time)
        self._complete_ready(
            now,
            fallback_position=fallback_position,
            fallback_velocity=fallback_velocity,
        )
        while self._next_start <= now + 1e-12:
            # The scheduler is sampled by the lower clock.  Floating-point
            # round-off can make an exactly due periodic event a few ulps later
            # than ``now``; the real computation starts in this current cycle,
            # never in its future.
            t_generate = min(self._next_start, now)
            self._next_start += self._period
            self.high_level_starts += 1
            version = self._next_version
            self._next_version += 1
            if self._rng.random() < float(self.config.skip_probability):
                self.high_level_skips += 1
                continue
            delay = float(self._rng.uniform(
                self.config.delay_min_seconds,
                self.config.delay_max_seconds,
            ))
            self._pending.append(
                _PendingUpperResult(
                    version=version,
                    t_generate=t_generate,
                    ready_at=t_generate + delay,
                    decision=decision_factory(t_generate),
                )
            )
            self.maximum_pending = max(self.maximum_pending, len(self._pending))
        # A zero-delay job is available in this same low-level control cycle.
        self._complete_ready(
            now,
            fallback_position=fallback_position,
            fallback_velocity=fallback_velocity,
        )

    def metrics(self) -> dict[str, float | int]:
        delay_values = np.asarray(self.delays, dtype=np.float64)
        switch_values = np.asarray(self.switching_errors, dtype=np.float64)
        return {
            "high_level_starts": int(self.high_level_starts),
            "high_level_completed": int(self.high_level_completed),
            "high_level_skips": int(self.high_level_skips),
            "commit_rejections": int(self.commit_rejections),
            "pending_packets": int(self.pending_count),
            "maximum_pending_packets": int(self.maximum_pending),
            "mean_compute_delay_seconds": float(delay_values.mean()) if delay_values.size else 0.0,
            "max_compute_delay_seconds": float(delay_values.max()) if delay_values.size else 0.0,
            "mean_reference_switching_error": float(switch_values.mean()) if switch_values.size else 0.0,
            "max_reference_switching_error": float(switch_values.max()) if switch_values.size else 0.0,
        }


class LowLevelReferenceSelector:
    """Non-blocking low-level packet reader with hover-hold fallback."""

    def __init__(self, initial_position):
        self.last_position = np.asarray(initial_position, dtype=np.float32).reshape(3).copy()
        self.last_velocity = np.zeros(3, dtype=np.float32)
        self.low_level_cycles = 0
        self.valid_packet_cycles = 0
        self.stale_packet_cycles = 0
        self.missing_packet_cycles = 0
        self.interruptions = 0
        self.packet_ages: list[float] = []

    def select(self, buffer: AsyncBuffer, current_time: float) -> tuple[ReferenceSample | None, bool]:
        """Return the current reference immediately; ``False`` means hover hold."""

        self.low_level_cycles += 1
        packet = buffer.latest_valid(current_time)
        if packet is not None:
            sample = packet.sample(current_time)
            self.last_position = sample.position.copy()
            self.last_velocity = sample.velocity.copy()
            self.valid_packet_cycles += 1
            self.packet_ages.append(sample.packet_age)
            return sample, False

        # Crucially, no wait or retry occurs here.  The PID gets a deterministic
        # hover reference every lower cycle even before the first packet arrives.
        self.last_velocity.fill(0.0)
        if buffer.latest_packet is None:
            self.missing_packet_cycles += 1
        else:
            self.stale_packet_cycles += 1
        return None, True

    def metrics(self) -> dict[str, float | int]:
        ages = np.asarray(self.packet_ages, dtype=np.float64)
        cycles = max(1, self.low_level_cycles)
        return {
            "low_level_cycles": int(self.low_level_cycles),
            "valid_packet_cycles": int(self.valid_packet_cycles),
            "stale_packet_cycles": int(self.stale_packet_cycles),
            "missing_packet_cycles": int(self.missing_packet_cycles),
            "stale_ratio": float(self.stale_packet_cycles / cycles),
            "low_level_interruptions": int(self.interruptions),
            "mean_packet_age_seconds": float(ages.mean()) if ages.size else 0.0,
            "p95_packet_age_seconds": float(np.percentile(ages, 95.0)) if ages.size else 0.0,
            "max_packet_age_seconds": float(ages.max()) if ages.size else 0.0,
        }
