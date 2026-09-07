import tempfile
import unittest

import numpy as np
import torch

from async_control import ReferencePacket
from high_level_planner import (
    FUTURE_TIMES,
    ReferenceSequenceDecision,
    ReferenceSequencePacketGenerator,
    ReferenceSequencePlanner,
    ReferenceTrajectoryDataset,
    extract_planner_input,
    reference_imitation_loss,
    save_reference_dataset,
)


class ReferencePlannerTest(unittest.TestCase):
    def test_network_uses_fixed_time_30d_output(self):
        planner = ReferenceSequencePlanner()
        output = planner(torch.zeros(4, 27))
        self.assertEqual(output.shape, (4, 30))
        sequence = planner.predict_sequence(np.zeros(27, dtype=np.float32))
        self.assertEqual(sequence.shape, (5, 6))
        loss, parts = reference_imitation_loss(output, torch.zeros(4, 5, 6))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("position_loss", parts)

    def test_extract_input_uses_goal_and_ranges_tail(self):
        observation = np.arange(263, dtype=np.float32)
        latent = np.arange(16, dtype=np.float32)
        value = extract_planner_input(latent, observation)
        np.testing.assert_array_equal(value[:16], latent)
        np.testing.assert_array_equal(value[16:], observation[-11:])

    def test_packet_generator_preserves_switch_and_fixed_times(self):
        old = ReferencePacket(
            version=1,
            t_generate=0.0,
            t_ready=0.0,
            valid_until=1.0,
            positions=np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=np.float32),
            velocities=np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
            relative_times=np.array([0.0, 1.0]),
            latent=np.zeros(16, dtype=np.float32),
        )
        decision = ReferenceSequenceDecision(
            position_offsets=np.stack(
                [np.array([0.05 * index, 0.0, 0.0]) for index in range(1, 6)]
            ),
            velocities=np.tile([0.5, 0.0, 0.0], (5, 1)),
            relative_times=FUTURE_TIMES,
            current_velocity=np.array([0.5, 0.0, 0.0]),
            latent=np.ones(16, dtype=np.float32),
        )
        generator = ReferenceSequencePacketGenerator()
        new = generator.generate(
            decision,
            version=2,
            t_generate=0.15,
            t_ready=0.2,
            previous_packet=old,
            fallback_position=np.zeros(3),
            fallback_velocity=np.zeros(3),
        )
        expected = old.sample(0.2)
        np.testing.assert_allclose(new.positions[0], expected.position, atol=1e-7)
        np.testing.assert_allclose(new.velocities[0], expected.velocity, atol=1e-7)
        np.testing.assert_allclose(new.relative_times, np.arange(6) * 0.1)
        self.assertEqual(new.positions.shape, (6, 3))
        self.assertEqual(new.latent.shape, (16,))
        self.assertLess(generator.last_switching_error, 1e-7)

    def test_dataset_has_episode_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as root:
            episode_ids = np.repeat(np.arange(6), 3)
            save_reference_dataset(
                root,
                episode_id=episode_ids,
                step_id=np.tile(np.arange(3), 6),
                planner_input=np.zeros((18, 27), dtype=np.float32),
                reference_target=np.random.default_rng(3).normal(size=(18, 5, 6)).astype(np.float32),
                metadata={"teacher_checkpoint": "test"},
                split_seed=4,
            )
            train = ReferenceTrajectoryDataset(root, "train")
            validation = ReferenceTrajectoryDataset(root, "validation")
            test = ReferenceTrajectoryDataset(root, "test")
            try:
                split_sets = [set(data.splits[name].tolist()) for data, name in (
                    (train, "train"),
                    (validation, "validation"),
                    (test, "test"),
                )]
                self.assertTrue(split_sets[0].isdisjoint(split_sets[1]))
                self.assertTrue(split_sets[0].isdisjoint(split_sets[2]))
                self.assertTrue(split_sets[1].isdisjoint(split_sets[2]))
                planner_input, target = train[0]
                self.assertEqual(planner_input.shape, (27,))
                self.assertEqual(target.shape, (5, 6))
            finally:
                train.close()
                validation.close()
                test.close()


if __name__ == "__main__":
    unittest.main()
