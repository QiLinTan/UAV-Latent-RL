import unittest

import numpy as np

from data.stratified_replay_buffer import DifficultyStratifiedReplayBuffer


class DifficultyStratifiedReplayBufferTest(unittest.TestCase):
    def test_sampling_balances_available_difficulty_levels(self):
        buffer = DifficultyStratifiedReplayBuffer(
            state_dim=2,
            action_dim=1,
            max_size=40,
            difficulties=(1, 2, 3, 5),
        )
        for difficulty, transition_count in ((1, 20), (2, 8), (3, 3), (5, 1)):
            for index in range(transition_count):
                state = np.array([difficulty, index], dtype=np.float32)
                buffer.push(
                    difficulty,
                    state,
                    np.array([0.0], dtype=np.float32),
                    1.0,
                    state,
                    False,
                )

        state, *_ = buffer.sample(12)
        sampled_levels, sampled_counts = np.unique(
            state[:, 0].numpy().astype(int), return_counts=True
        )
        self.assertEqual(dict(zip(sampled_levels, sampled_counts)), {1: 3, 2: 3, 3: 3, 5: 3})
        self.assertEqual(buffer.last_sample_counts, {1: 3, 2: 3, 3: 3, 5: 3})


if __name__ == "__main__":
    unittest.main()
