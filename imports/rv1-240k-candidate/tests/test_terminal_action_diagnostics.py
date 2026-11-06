import unittest

import numpy as np

from trainers.terminal_action_diagnostics import (
    summarize_action_interface,
    summarize_xy_q_scan,
    xy_action_candidates,
)


class TerminalActionDiagnosticsTest(unittest.TestCase):
    def test_xy_scan_uses_requested_physical_speeds_and_preserves_z(self):
        actions, metadata = xy_action_candidates(
            np.array([0.0, 1.0, -0.4], dtype=np.float32),
            position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            max_planar_speed=0.8,
        )
        self.assertEqual(actions.shape, (12, 3))
        np.testing.assert_allclose(actions[:, 2], -0.4)
        goal_rows = [
            index
            for index, item in enumerate(metadata)
            if item["kind"] == "speed_scan" and item["direction"] == "goal"
        ]
        np.testing.assert_allclose(
            np.linalg.norm(actions[goal_rows, :2], axis=1) * 0.8,
            [0.1, 0.2, 0.4, 0.6, 0.8],
            atol=1e-6,
        )
        actor_row = next(
            index
            for index, item in enumerate(metadata)
            if item["kind"] == "speed_scan"
            and item["direction"] == "actor"
            and abs(item["speed_mps"] - 0.4) < 1e-6
        )
        np.testing.assert_allclose(actions[actor_row, :2], [0.0, 0.5], atol=1e-6)

    def test_action_interface_summary_detects_linear_mapping(self):
        records = []
        for norm in (0.1, 0.3, 0.5, 1.1):
            records.append(
                {
                    "episode": 4,
                    "actor_xy_norm": norm,
                    "cmd_xy_speed": min(norm, 1.0) * 0.8,
                    "actual_xy_speed": 0.4,
                }
            )
        summary = summarize_action_interface(records, 0.8)
        self.assertTrue(summary["mapping_is_linear_below_unit_norm"])
        self.assertAlmostEqual(summary["mapping_max_abs_error"], 0.0)

    def test_q_summary_uses_conservative_candidate_values(self):
        records = [
            {
                "episode": 4,
                "candidates": [
                    {
                        "kind": "speed_scan",
                        "direction": "stop",
                        "speed_mps": 0.0,
                        "q_min": 2.0,
                    },
                    {
                        "kind": "speed_scan",
                        "direction": "goal",
                        "speed_mps": 0.8,
                        "q_min": 1.0,
                    },
                    {
                        "kind": "actor_raw",
                        "direction": "actor_raw",
                        "speed_mps": 0.8,
                        "q_min": 0.5,
                    },
                ],
            }
        ]
        summary = summarize_xy_q_scan(records)
        self.assertEqual(summary["best_candidate_counts"], {"stop:0.0": 1})
        self.assertAlmostEqual(summary["mean_best_minus_actor_q"], 1.5)


if __name__ == "__main__":
    unittest.main()
