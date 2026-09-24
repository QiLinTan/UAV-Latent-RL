import unittest

import numpy as np

from trainers.timeout_diagnostics import classify_timeout


def _samples(command=(0.4, 0.0, 0.0), actual=(0.4, 0.0, 0.0)):
    return [
        {"cmd_velocity": command, "actual_velocity": actual, "clearance": 0.7}
        for _ in range(30)
    ]


class TimeoutDiagnosticsTest(unittest.TestCase):
    def classify(self, positions, goals, **kwargs):
        return classify_timeout(
            positions=np.asarray(positions, dtype=np.float32),
            goal_distances=np.asarray(goals, dtype=np.float32),
            target_position=np.array([3.5, 0.0, 1.0], dtype=np.float32),
            navigation_metrics={"backtrack_ratio": 0.1},
            last_window_samples=kwargs.pop("samples", _samples()),
            **kwargs,
        )["code"]

    def test_t3_xy_close_but_height_not_converged(self):
        self.assertEqual(
            self.classify([[0, 0, 1], [3.4, 0.1, 1.5]], [3.5, 0.52]), "T3"
        )

    def test_t4_low_motion(self):
        self.assertEqual(
            self.classify(
                [[0, 0, 1], [1, 0, 1]],
                [3.5, 2.5],
                samples=_samples((0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
            ),
            "T4",
        )

    def test_t1_near_goal_then_timeout(self):
        self.assertEqual(
            self.classify([[0, 0, 1], [3.1, 0.0, 1]], [3.5, 0.4]), "T1"
        )

    def test_t2_never_reaches_final_region(self):
        self.assertEqual(
            self.classify([[0, 0, 1], [1.5, 0, 1]], [3.5, 2.0]), "T2"
        )


if __name__ == "__main__":
    unittest.main()
