import unittest

import numpy as np

from models.continuous_world_reference import (
    ContinuousWorldPacketSource,
    TrapezoidalWorldTrajectory,
)
from models.reference_packet import AsyncReferenceBuffer, ReferencePacket
from scripts.validate_stage1_reference_pid import (
    Stage1Config,
    derive_timing,
)


class Stage1ContinuousWorldReferenceTest(unittest.TestCase):
    def setUp(self):
        self.config = Stage1Config()
        self.trajectory = TrapezoidalWorldTrajectory(
            self.config.start_position,
            self.config.target_position,
            max_speed=self.config.max_speed,
            max_acceleration=self.config.max_acceleration,
        )

    def test_seven_meter_trapezoid_obeys_endpoints_and_limits(self):
        times = np.linspace(0.0, self.trajectory.duration, 10_001)
        positions, velocities = self.trajectory.sample(times)

        np.testing.assert_allclose(positions[0], self.config.start_position, atol=1e-7)
        np.testing.assert_allclose(positions[-1], self.config.target_position, atol=1e-7)
        np.testing.assert_allclose(velocities[[0, -1]], 0.0, atol=1e-7)
        self.assertLessEqual(
            float(np.linalg.norm(velocities, axis=1).max()),
            self.config.max_speed + 1e-6,
        )
        dt = float(times[1] - times[0])
        acceleration = np.diff(velocities, axis=0) / dt
        self.assertLessEqual(
            float(np.linalg.norm(acceleration, axis=1).max()),
            self.config.max_acceleration + 1e-3,
        )

    def test_sliding_packets_are_world_frame_and_do_not_restart(self):
        source = ContinuousWorldPacketSource(self.trajectory)
        first = source.make_packet(0.0)
        next_time = 1.0 / self.config.high_frequency_hz
        second = source.make_packet(next_time)

        self.assertIsInstance(first, ReferencePacket)
        self.assertEqual(first.frame_id, "world")
        self.assertEqual(second.version, first.version + 1)
        expected_position, expected_velocity = self.trajectory.sample(next_time)
        np.testing.assert_allclose(second.positions[0], expected_position, atol=1e-7)
        np.testing.assert_allclose(second.velocities[0], expected_velocity, atol=1e-7)

        old_sample = first.sample(next_time)
        np.testing.assert_allclose(
            second.positions[0], old_sample["current_position"], atol=2e-5
        )
        np.testing.assert_allclose(
            second.velocities[0], old_sample["current_velocity"], atol=2e-5
        )
        self.assertGreater(float(second.velocities[0, 0]), 0.0)

    def test_buffer_interpolation_matches_global_trajectory(self):
        source = ContinuousWorldPacketSource(self.trajectory)
        buffer = AsyncReferenceBuffer()
        packet = source.make_packet(2.0)
        self.assertTrue(buffer.publish(packet))

        query_time = 2.037
        sample = buffer.sample(query_time)
        expected_position, expected_velocity = self.trajectory.sample(query_time)
        # Linear packet interpolation approximates the continuous parabola.
        np.testing.assert_allclose(
            sample["current_position"], expected_position, atol=1e-3
        )
        np.testing.assert_allclose(
            sample["current_velocity"], expected_velocity, atol=1e-3
        )
        self.assertTrue(sample["valid"])

    def test_packet_versions_are_strictly_continuous(self):
        source = ContinuousWorldPacketSource(self.trajectory)
        buffer = AsyncReferenceBuffer()
        versions = []
        for index in range(20):
            packet = source.make_packet(index / self.config.high_frequency_hz)
            self.assertTrue(buffer.publish(packet))
            versions.append(buffer.version)
        self.assertEqual(versions, list(range(1, 21)))
        self.assertEqual(buffer.rejected_packets, 0)

    def test_timeout_is_derived_from_trajectory_and_reachable(self):
        timing = derive_timing(self.config, self.trajectory)
        self.assertTrue(timing["reachability_assertion"])
        self.assertAlmostEqual(
            timing["timeout"],
            timing["trajectory_duration"] + self.config.settle_seconds,
        )
        self.assertGreaterEqual(
            timing["trajectory_duration"],
            timing["physical_reachability_lower_bound"],
        )


if __name__ == "__main__":
    unittest.main()
