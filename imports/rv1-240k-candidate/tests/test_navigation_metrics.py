import unittest

import numpy as np

from trainers.navigation_metrics import episode_navigation_metrics


class EpisodeNavigationMetricsTest(unittest.TestCase):
    def test_straight_progress_is_efficient_without_backtracking(self):
        metrics = episode_navigation_metrics(
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            goal_distances=[2, 1, 0],
            timestep_seconds=1.0,
        )
        self.assertAlmostEqual(metrics["path_length"], 2.0)
        self.assertAlmostEqual(metrics["path_efficiency"], 1.0)
        self.assertAlmostEqual(metrics["backtrack_ratio"], 0.0)
        self.assertAlmostEqual(metrics["max_no_progress_seconds"], 0.0)
        self.assertAlmostEqual(metrics["stuck_window_fraction"], 0.0)
        self.assertAlmostEqual(metrics["revisit_rate"], 0.0)

    def test_backtracking_and_stalling_are_measured_from_goal_distance(self):
        metrics = episode_navigation_metrics(
            positions=[[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
            goal_distances=[3, 2, 3, 2],
            timestep_seconds=1.0,
            stuck_window_seconds=2.0,
        )
        self.assertAlmostEqual(metrics["path_length"], 3.0)
        self.assertAlmostEqual(metrics["path_efficiency"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["backtrack_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["max_no_progress_seconds"], 2.0)
        self.assertAlmostEqual(metrics["stuck_window_fraction"], 0.5)
        self.assertGreater(metrics["revisit_rate"], 0.0)

    def test_spatial_cell_revisits_are_counted(self):
        metrics = episode_navigation_metrics(
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.0, 0.0]]
            ),
            goal_distances=[1.0, 0.9, 0.8, 0.4],
            timestep_seconds=0.5,
        )
        self.assertAlmostEqual(metrics["revisit_rate"], 2.0 / 3.0)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            episode_navigation_metrics(
                positions=[[0, 0, 0]],
                goal_distances=[1.0, 0.0],
                timestep_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
