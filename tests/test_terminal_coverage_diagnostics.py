import unittest

import numpy as np

from trainers.terminal_coverage_diagnostics import (
    summarize_terminal_coverage,
    summarize_terminal_episode,
)


class TerminalCoverageDiagnosticsTest(unittest.TestCase):
    def test_counts_guidance_entry_and_success_crossing(self):
        positions = np.array(
            [
                [2.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.7, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ]
        )
        velocities = np.tile([-0.4, 0.0, 0.0], (5, 1))
        result = summarize_terminal_episode(
            positions=positions,
            velocities=velocities,
            target_position=[0.0, 0.0, 0.0],
            goal_tolerance=0.2,
            guidance_radius=1.2,
            done_reason="success",
            timestep_seconds=0.1,
        )
        self.assertTrue(result["entered_guidance_region"])
        self.assertEqual(result["guidance_entry_count"], 1)
        self.assertEqual(result["guidance_state_count"], 3)
        self.assertEqual(result["successful_terminal_transition_count"], 1)
        self.assertAlmostEqual(result["guidance_mean_closing_speed_mps"], 0.4)
        self.assertAlmostEqual(result["guidance_away_motion_fraction"], 0.0)

    def test_separates_xy_proximity_from_3d_guidance_coverage(self):
        positions = np.array(
            [
                [2.0, 0.0, 1.5],
                [0.8, 0.0, 1.5],
                [0.4, 0.0, 1.5],
            ]
        )
        velocities = np.zeros_like(positions)
        result = summarize_terminal_episode(
            positions=positions,
            velocities=velocities,
            target_position=[0.0, 0.0, 0.0],
            goal_tolerance=0.2,
            guidance_radius=1.2,
            done_reason="timeout",
            timestep_seconds=0.1,
        )
        self.assertFalse(result["entered_guidance_region"])
        self.assertEqual(result["guidance_state_count"], 0)
        self.assertEqual(result["xy_close_but_3d_outside_state_count"], 2)

    def test_aggregate_uses_episode_and_transition_denominators(self):
        first = {
            "entered_guidance_region": True,
            "successful_terminal_transition_count": 1,
            "transition_count": 4,
            "guidance_state_count": 2,
            "minimum_goal_distance_m": 0.1,
            "xy_close_but_3d_outside_state_count": 0,
            "distance_bin_state_counts": {"[0.2,0.4)": 1},
        }
        second = {
            "entered_guidance_region": False,
            "successful_terminal_transition_count": 0,
            "transition_count": 6,
            "guidance_state_count": 0,
            "minimum_goal_distance_m": 1.4,
            "xy_close_but_3d_outside_state_count": 3,
            "distance_bin_state_counts": {"[0.2,0.4)": 0},
        }
        summary = summarize_terminal_coverage([first, second])
        self.assertEqual(summary["guidance_entered_episode_count"], 1)
        self.assertAlmostEqual(summary["guidance_state_fraction"], 0.2)
        self.assertEqual(summary["successful_terminal_transition_count"], 1)


if __name__ == "__main__":
    unittest.main()
