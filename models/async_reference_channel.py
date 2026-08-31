from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.reference_packet import AsyncReferenceBuffer, ReferencePacket


@dataclass(frozen=True)
class AsyncChannelProfile:
    """Deterministic, replayable high-to-low communication fault profile."""

    name: str
    upper_frequency_hz: float
    fixed_latency_seconds: float = 0.0
    latency_jitter_seconds: float = 0.0
    drop_probability: float = 0.0
    burst_drop_windows: tuple[tuple[float, float], ...] = ()
    duplicate_every_n: int = 0
    reorder_every_n: int = 0
    reorder_extra_delay_seconds: float = 0.0
    seed: int = 0
    expected_to_expire: bool = False
    extreme: bool = False

    def __post_init__(self):
        if float(self.upper_frequency_hz) <= 0.0:
            raise ValueError("upper_frequency_hz must be positive.")
        if float(self.fixed_latency_seconds) < 0.0:
            raise ValueError("fixed_latency_seconds cannot be negative.")
        if float(self.latency_jitter_seconds) < 0.0:
            raise ValueError("latency_jitter_seconds cannot be negative.")
        if not 0.0 <= float(self.drop_probability) <= 1.0:
            raise ValueError("drop_probability must be in [0, 1].")
        if int(self.duplicate_every_n) < 0 or int(self.reorder_every_n) < 0:
            raise ValueError("duplicate_every_n and reorder_every_n cannot be negative.")
        if float(self.reorder_extra_delay_seconds) < 0.0:
            raise ValueError("reorder_extra_delay_seconds cannot be negative.")
        for start, end in self.burst_drop_windows:
            if float(start) < 0.0 or float(end) <= float(start):
                raise ValueError("Each burst window must satisfy 0 <= start < end.")


class AsyncReferenceChannel:
    """Queue ReferencePackets through repeatable latency and packet faults."""

    def __init__(self, profile: AsyncChannelProfile):
        self.profile = profile
        self._rng = np.random.default_rng(int(profile.seed))
        self._queue: list[tuple[float, int, ReferencePacket]] = []
        self._submit_count = 0
        self._queue_order = 0
        self.generated_packets = 0
        self.dropped_packets = 0
        self.random_drops = 0
        self.burst_drops = 0
        self.duplicated_packets = 0
        self.reordered_packets = 0
        self.delivered_packets = 0
        self.accepted_deliveries = 0
        self.rejected_deliveries = 0
        self.maximum_queue_depth = 0

    @property
    def pending_packets(self) -> int:
        return len(self._queue)

    def _inside_burst(self, now: float) -> bool:
        return any(
            float(start) <= float(now) < float(end)
            for start, end in self.profile.burst_drop_windows
        )

    def submit(self, packet: ReferencePacket, now: float) -> bool:
        if not isinstance(packet, ReferencePacket):
            raise TypeError("AsyncReferenceChannel accepts ReferencePacket values only.")
        self._submit_count += 1
        self.generated_packets += 1
        if self._inside_burst(now):
            self.dropped_packets += 1
            self.burst_drops += 1
            return False
        if self._rng.random() < float(self.profile.drop_probability):
            self.dropped_packets += 1
            self.random_drops += 1
            return False

        jitter = self._rng.uniform(
            -float(self.profile.latency_jitter_seconds),
            float(self.profile.latency_jitter_seconds),
        )
        latency = max(0.0, float(self.profile.fixed_latency_seconds) + jitter)
        if (
            int(self.profile.reorder_every_n) > 0
            and self._submit_count % int(self.profile.reorder_every_n) == 0
        ):
            latency += float(self.profile.reorder_extra_delay_seconds)
            self.reordered_packets += 1
        arrival = float(now) + latency
        self._enqueue(packet.with_receive_time(arrival), arrival)

        if (
            int(self.profile.duplicate_every_n) > 0
            and self._submit_count % int(self.profile.duplicate_every_n) == 0
        ):
            duplicate_arrival = arrival + 1e-6
            self._enqueue(packet.with_receive_time(duplicate_arrival), duplicate_arrival)
            self.duplicated_packets += 1
        return True

    def _enqueue(self, packet: ReferencePacket, arrival: float):
        self._queue_order += 1
        self._queue.append((float(arrival), self._queue_order, packet))
        self.maximum_queue_depth = max(self.maximum_queue_depth, len(self._queue))

    def deliver(self, now: float, buffer: AsyncReferenceBuffer) -> list[dict]:
        ready = [item for item in self._queue if item[0] <= float(now) + 1e-12]
        self._queue = [item for item in self._queue if item[0] > float(now) + 1e-12]
        ready.sort(key=lambda item: (item[0], item[1]))
        events = []
        for arrival, _, packet in ready:
            accepted = bool(buffer.publish(packet))
            self.delivered_packets += 1
            self.accepted_deliveries += int(accepted)
            self.rejected_deliveries += int(not accepted)
            events.append(
                {
                    "arrival_time": float(arrival),
                    "version": int(packet.version),
                    "accepted": accepted,
                }
            )
        return events

    def metrics(self) -> dict:
        return {
            "generated_packets": int(self.generated_packets),
            "dropped_packets": int(self.dropped_packets),
            "random_drops": int(self.random_drops),
            "burst_drops": int(self.burst_drops),
            "duplicated_packets": int(self.duplicated_packets),
            "reordered_packets": int(self.reordered_packets),
            "delivered_packets": int(self.delivered_packets),
            "accepted_deliveries": int(self.accepted_deliveries),
            "rejected_deliveries": int(self.rejected_deliveries),
            "pending_packets": int(self.pending_packets),
            "maximum_queue_depth": int(self.maximum_queue_depth),
        }
