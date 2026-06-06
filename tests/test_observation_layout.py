import unittest

import numpy as np
import torch

from algos.td3.td3_latent_affordance import TD3LatentAffordance
from envs.observation_layout import (
    GOAL_DIM,
    KIN_DIM,
    RANGE_DIM,
    ForestObservationLayout,
    append_forest_task_observation,
)


class ForestObservationLayoutTest(unittest.TestCase):
    def test_current_263d_layout_uses_tail_for_goal_and_ranges(self):
        layout = ForestObservationLayout.from_total_dim(263, action_dim=4)
        observation = torch.arange(263, dtype=torch.float32).reshape(1, -1)

        parts = layout.split(observation)

        self.assertEqual(parts.kin.shape[-1], KIN_DIM)
        self.assertEqual(parts.action_history.shape[-1], 240)
        self.assertEqual(parts.goal.shape[-1], GOAL_DIM)
        self.assertEqual(parts.ranges.shape[-1], RANGE_DIM)
        self.assertTrue(torch.equal(parts.goal, observation[:, 252:255]))
        self.assertTrue(torch.equal(parts.ranges, observation[:, 255:263]))

    def test_layout_adapts_to_shorter_action_history(self):
        layout = ForestObservationLayout.from_total_dim(95, action_dim=4)
        self.assertEqual(layout.action_history_dim, 72)
        self.assertEqual(layout.goal_start, 84)
        self.assertEqual(layout.range_start, 87)

    def test_append_validates_and_preserves_order(self):
        base = np.zeros((2, 20), dtype=np.float32)
        goal = np.full((2, GOAL_DIM), 2.0, dtype=np.float32)
        ranges = np.full((2, RANGE_DIM), 3.0, dtype=np.float32)

        observation = append_forest_task_observation(base, goal, ranges)
        parts = ForestObservationLayout.from_total_dim(observation.shape[-1], action_dim=4).split(
            observation
        )

        np.testing.assert_array_equal(parts.goal, goal)
        np.testing.assert_array_equal(parts.ranges, ranges)

    def test_rejects_misaligned_action_history(self):
        with self.assertRaises(ValueError):
            ForestObservationLayout.from_total_dim(264, action_dim=4)

    def test_affordance_range_target_reads_tail_not_action_history(self):
        agent = TD3LatentAffordance(
            state_dim=263,
            action_dim=4,
            max_action=1.0,
            device=torch.device("cpu"),
        )
        observation = torch.zeros((2, 263), dtype=torch.float32)
        observation[:, 15:23] = -1.0
        observation[0, 255:263] = 0.25
        observation[1, 255:263] = 0.75

        actual = agent._range_min(observation)

        expected = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
