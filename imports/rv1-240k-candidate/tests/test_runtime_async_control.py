import multiprocessing as mp
import queue
import time
import unittest

import numpy as np

from async_control.latest_channel import get_latest, put_latest
from async_control.latest_channel import LatestObservationSlot
from async_control.runtime_high_level_worker import RuntimeHighLevelWorker, RuntimeWorkerConfig


def snapshot(index=0):
    return {
        "sim_time": index / 120.0,
        "wall_time": time.perf_counter(),
        "observation": np.zeros(263, dtype=np.float32),
        "position": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "yaw": 0.0,
        "target_position": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        "fallback_position": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "fallback_velocity": np.zeros(3, dtype=np.float32),
        "previous_packet": None,
    }


def wait_for(predicate, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    return None


class RuntimeAsyncControlTest(unittest.TestCase):
    def test_latest_only_observation_channel(self):
        channel = queue.Queue(maxsize=1)
        put_latest(channel, {"version": 1})
        put_latest(channel, {"version": 2})
        put_latest(channel, {"version": 3})
        self.assertEqual(get_latest(channel)["version"], 3)
        self.assertIsNone(get_latest(channel))

    def test_shared_observation_slot_is_latest_only(self):
        ctx = mp.get_context("spawn")
        slot = LatestObservationSlot(ctx)
        slot.put(snapshot(1))
        first = slot.get_latest(0)
        self.assertEqual(first["version"], 1)
        slot.put(snapshot(2))
        slot.put(snapshot(3))
        self.assertIsNone(slot.get_latest(3))
        latest = slot.get_latest(1)
        self.assertEqual(latest["version"], 3)
        self.assertAlmostEqual(latest["sim_time"], 3 / 120.0)

    def test_latest_only_packet_channel(self):
        channel = queue.Queue(maxsize=1)
        put_latest(channel, {"packet": "old"})
        put_latest(channel, {"packet": "new"})
        self.assertEqual(get_latest(channel)["packet"], "new")

    def test_worker_starts_and_stops_without_zombie(self):
        worker = RuntimeHighLevelWorker(RuntimeWorkerConfig(policy_mode="constant"))
        worker.start()
        try:
            started = wait_for(lambda: [event for event in worker.get_all_status() if event.get("type") == "started"])
            self.assertTrue(started)
        finally:
            stop = worker.stop()
        self.assertFalse(stop["alive"])
        self.assertIsNotNone(stop["exitcode"])

    def test_worker_exception_does_not_block_low_level_reads(self):
        worker = RuntimeHighLevelWorker(RuntimeWorkerConfig(policy_mode="crash_after_snapshot"))
        worker.start()
        try:
            wait_for(lambda: [event for event in worker.get_all_status() if event.get("type") == "started"])
            worker.publish_observation(snapshot())
            error = wait_for(lambda: [event for event in worker.get_all_status() if event.get("type") == "error"])
            self.assertTrue(error)
            cycles = 0
            deadline = time.perf_counter() + 0.05
            while time.perf_counter() < deadline:
                worker.get_packet()
                cycles += 1
            self.assertGreater(cycles, 0)
        finally:
            stop = worker.stop()
        self.assertFalse(stop["alive"])

    def test_packet_versions_do_not_regress(self):
        worker = RuntimeHighLevelWorker(
            RuntimeWorkerConfig(policy_mode="constant", high_frequency_hz=1000.0)
        )
        worker.start()
        versions = []
        try:
            wait_for(lambda: [event for event in worker.get_all_status() if event.get("type") == "started"])
            for index in range(3):
                worker.publish_observation(snapshot(index))
                message = wait_for(lambda: worker.get_packet())
                self.assertIsNotNone(message)
                versions.append(message["packet"].version)
                worker.get_all_status()
        finally:
            worker.stop()
        self.assertEqual(versions, sorted(versions))
        self.assertEqual(len(set(versions)), len(versions))

    def test_high_level_100ms_delay_does_not_stop_low_level_loop(self):
        worker = RuntimeHighLevelWorker(
            RuntimeWorkerConfig(
                policy_mode="constant",
                high_frequency_hz=1000.0,
                artificial_delay_seconds=0.100,
            )
        )
        worker.start()
        try:
            wait_for(lambda: [event for event in worker.get_all_status() if event.get("type") == "started"])
            worker.publish_observation(snapshot())
            cycles = 0
            saw_packet_early = False
            deadline = time.perf_counter() + 0.05
            while time.perf_counter() < deadline:
                saw_packet_early = saw_packet_early or worker.get_packet() is not None
                cycles += 1
            self.assertGreater(cycles, 0)
            self.assertFalse(saw_packet_early)
            message = wait_for(lambda: worker.get_packet(), timeout=2.0)
            self.assertIsNotNone(message)
            self.assertGreaterEqual(message["worker_timing"]["artificial_delay_time_seconds"], 0.100)
        finally:
            stop = worker.stop()
        self.assertFalse(stop["alive"])


if __name__ == "__main__":
    unittest.main()
