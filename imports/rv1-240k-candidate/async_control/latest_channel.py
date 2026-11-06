"""Latest-only helpers for multiprocessing queues."""

from __future__ import annotations

import queue

import numpy as np


def put_latest(channel, value) -> None:
    """Put ``value`` while discarding any older queued value.

    ``multiprocessing.Queue(maxsize=1)`` is intentionally used as a small IPC
    primitive here.  The maxsize check is not a correctness boundary, so the
    helper also drains stale items before retrying.
    """

    while True:
        try:
            channel.put_nowait(value)
            return
        except queue.Full:
            try:
                channel.get_nowait()
            except queue.Empty:
                pass


def get_latest(channel):
    """Return the newest queued value, or ``None`` when the channel is empty."""

    latest = None
    while True:
        try:
            latest = channel.get_nowait()
        except queue.Empty:
            return latest


class LatestObservationSlot:
    """Shared-memory latest observation snapshot for high-rate low-to-high IPC."""

    def __init__(self, context, observation_dim: int = 263):
        self.observation_dim = int(observation_dim)
        self.lock = context.Lock()
        self.version = context.Value("Q", 0)
        self.sim_time = context.Value("d", 0.0)
        self.wall_time = context.Value("d", 0.0)
        self.yaw = context.Value("d", 0.0)
        self.observation = context.Array("d", self.observation_dim, lock=False)
        self.position = context.Array("d", 3, lock=False)
        self.velocity = context.Array("d", 3, lock=False)
        self.target_position = context.Array("d", 3, lock=False)
        self.fallback_position = context.Array("d", 3, lock=False)
        self.fallback_velocity = context.Array("d", 3, lock=False)

    def put(self, snapshot: dict) -> int:
        observation = np.asarray(snapshot["observation"], dtype=np.float64).reshape(self.observation_dim)
        position = np.asarray(snapshot["position"], dtype=np.float64).reshape(3)
        velocity = np.asarray(snapshot["velocity"], dtype=np.float64).reshape(3)
        target = np.asarray(snapshot["target_position"], dtype=np.float64).reshape(3)
        fallback_position = np.asarray(snapshot["fallback_position"], dtype=np.float64).reshape(3)
        fallback_velocity = np.asarray(snapshot["fallback_velocity"], dtype=np.float64).reshape(3)
        with self.lock:
            np.frombuffer(self.observation, dtype=np.float64)[:] = observation
            np.frombuffer(self.position, dtype=np.float64)[:] = position
            np.frombuffer(self.velocity, dtype=np.float64)[:] = velocity
            np.frombuffer(self.target_position, dtype=np.float64)[:] = target
            np.frombuffer(self.fallback_position, dtype=np.float64)[:] = fallback_position
            np.frombuffer(self.fallback_velocity, dtype=np.float64)[:] = fallback_velocity
            self.sim_time.value = float(snapshot["sim_time"])
            self.wall_time.value = float(snapshot["wall_time"])
            self.yaw.value = float(snapshot["yaw"])
            self.version.value += 1
            return int(self.version.value)

    def get_latest(self, last_version: int = 0) -> dict | None:
        with self.lock:
            version = int(self.version.value)
            if version <= int(last_version):
                return None
            return {
                "version": version,
                "sim_time": float(self.sim_time.value),
                "wall_time": float(self.wall_time.value),
                "yaw": float(self.yaw.value),
                "observation": np.frombuffer(self.observation, dtype=np.float64).astype(np.float32).copy(),
                "position": np.frombuffer(self.position, dtype=np.float64).astype(np.float32).copy(),
                "velocity": np.frombuffer(self.velocity, dtype=np.float64).astype(np.float32).copy(),
                "target_position": np.frombuffer(self.target_position, dtype=np.float64).astype(np.float32).copy(),
                "fallback_position": np.frombuffer(self.fallback_position, dtype=np.float64).astype(np.float32).copy(),
                "fallback_velocity": np.frombuffer(self.fallback_velocity, dtype=np.float64).astype(np.float32).copy(),
                "previous_packet": None,
            }
