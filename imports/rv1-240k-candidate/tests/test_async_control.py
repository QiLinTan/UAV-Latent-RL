import unittest

import numpy as np

from async_control import (
    AsyncBuffer,
    AsyncScheduler,
    AsyncSchedulerConfig,
    HighLevelDecision,
    LowLevelReferenceSelector,
    ReferenceGenerator,
    ReferencePacket,
)


def packet(version=1, *, ready=0.0, valid_until=1.0):
    return ReferencePacket(
        version=version,
        t_generate=0.0,
        t_ready=ready,
        valid_until=valid_until,
        positions=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        relative_times=np.array([0.0, 1.0]),
        latent=np.zeros(16, dtype=np.float32),
    )


class AsyncControlTest(unittest.TestCase):
    def test_buffer_keeps_newest_valid_packet_and_age(self):
        buffer = AsyncBuffer()
        self.assertIsNone(buffer.latest_valid(0.0))
        self.assertTrue(buffer.commit(packet(2)))
        self.assertFalse(buffer.commit(packet(1)))
        self.assertEqual(buffer.latest_valid(0.25).version, 2)
        self.assertAlmostEqual(buffer.packet_age(0.25), 0.25)
        self.assertIsNone(buffer.latest_valid(1.01))

    def test_generator_switch_starts_at_old_reference(self):
        old = packet(1, ready=0.0, valid_until=1.0)
        generator = ReferenceGenerator(horizon_seconds=0.5, sequence_length=8)
        new = generator.generate(
            HighLevelDecision(
                target_position=np.array([3.5, 0.0, 1.0]),
                desired_velocity=np.array([0.4, 0.0, 0.0]),
                latent=np.ones(16, dtype=np.float32),
            ),
            version=2,
            t_generate=0.3,
            t_ready=0.3,
            previous_packet=old,
            fallback_position=np.zeros(3),
            fallback_velocity=np.zeros(3),
        )
        expected = old.sample(0.3)
        np.testing.assert_allclose(new.positions[0], expected.position, atol=1e-7)
        np.testing.assert_allclose(new.velocities[0], expected.velocity, atol=1e-7)
        self.assertLess(generator.last_switching_error, 1e-7)
        self.assertLess(generator.last_velocity_switching_error, 1e-7)

    def test_delayed_upper_never_interrupts_lower_loop(self):
        buffer = AsyncBuffer()
        generator = ReferenceGenerator(horizon_seconds=0.1, sequence_length=4)
        scheduler = AsyncScheduler(
            AsyncSchedulerConfig(
                high_frequency_hz=15.0,
                delay_min_seconds=0.05,
                delay_max_seconds=0.05,
                skip_probability=0.1,
                seed=3,
            ),
            generator,
            buffer,
        )
        lower = LowLevelReferenceSelector([0.0, 0.0, 1.0])

        def decision(_):
            return HighLevelDecision(
                target_position=np.array([1.0, 0.0, 1.0]),
                desired_velocity=np.array([0.3, 0.0, 0.0]),
                latent=np.zeros(16, dtype=np.float32),
            )

        for step in range(60):
            now = step / 120.0
            scheduler.advance(
                now,
                decision,
                fallback_position=lower.last_position,
                fallback_velocity=np.zeros(3),
            )
            _, fallback = lower.select(buffer, now)
            self.assertIsInstance(fallback, bool)

        self.assertEqual(lower.metrics()["low_level_cycles"], 60)
        self.assertEqual(lower.metrics()["low_level_interruptions"], 0)
        self.assertGreater(scheduler.metrics()["high_level_starts"], 0)
        self.assertGreater(lower.metrics()["missing_packet_cycles"], 0)

    def test_expired_packet_uses_hold_position_and_zero_velocity(self):
        buffer = AsyncBuffer()
        buffer.commit(packet(1, ready=0.0, valid_until=0.1))
        lower = LowLevelReferenceSelector([0.0, 0.0, 1.0])
        sample, fallback = lower.select(buffer, 0.05)
        self.assertFalse(fallback)
        expected_position = sample.position.copy()

        sample, fallback = lower.select(buffer, 0.11)
        self.assertIsNone(sample)
        self.assertTrue(fallback)
        np.testing.assert_allclose(lower.last_position, expected_position)
        np.testing.assert_allclose(lower.last_velocity, np.zeros(3))
        self.assertEqual(lower.metrics()["stale_packet_cycles"], 1)
        self.assertEqual(lower.metrics()["low_level_interruptions"], 0)


if __name__ == "__main__":
    unittest.main()
