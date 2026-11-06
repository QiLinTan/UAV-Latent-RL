import unittest

from trainers.recurrent_diagnostics import (
    recurrent_observation_contract,
    replay_buffer_memory_bytes,
)


class RecurrentDiagnosticsTest(unittest.TestCase):
    def test_96_ray_history_contracts(self):
        for history_length, state_dim in ((4, 517), (8, 917), (16, 1717)):
            contract = recurrent_observation_contract(
                base_state_dim=117,
                history_frame_dim=100,
                history_length=history_length,
                state_dim=state_dim,
            )
            self.assertEqual(contract["expected_state_dim"], state_dim)

    def test_rejects_mismatched_contract(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            recurrent_observation_contract(
                base_state_dim=117,
                history_frame_dim=100,
                history_length=4,
                state_dim=516,
            )

    def test_replay_allocation_matches_float32_arrays(self):
        self.assertEqual(
            replay_buffer_memory_bytes(state_dim=517, action_dim=3, capacity=60_000),
            249_360_000,
        )


if __name__ == "__main__":
    unittest.main()
