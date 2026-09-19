import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trainers.upper_trajectory_export import export_upper_trajectory


class UpperTrajectoryExportTest(unittest.TestCase):
    def test_writes_requested_trace_plots_and_summary(self):
        actual = np.array(
            [[-3.5, 0.0, 1.0], [-3.0, 0.1, 1.05], [-2.5, 0.2, 1.0]],
            dtype=np.float32,
        )
        trace = {
            "time_seconds": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "actual_position": actual,
            "actual_velocity": np.zeros_like(actual),
            "reference_position": actual + np.array([0.05, 0.0, 0.0], dtype=np.float32),
            "reference_velocity": np.zeros_like(actual),
            "packet_endpoint": actual + np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "upper_action": np.zeros((2, 3), dtype=np.float32),
            "reward": np.array([0.1, 0.2], dtype=np.float32),
            "goal_distance": np.array([6.5, 6.0], dtype=np.float32),
            "min_tree_clearance": np.array([1.0, 0.8], dtype=np.float32),
            "packet_version": np.array([1, 2], dtype=np.int64),
            "packet_valid": np.array([True, True]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = export_upper_trajectory(
                tmp,
                seed=0,
                training_step=12000,
                trace=trace,
                tree_specs=[{"xy": [0.0, 0.2], "radius": 0.15, "height": 2.0}],
                start_position=[-3.5, 0.0, 1.0],
                target_position=[3.5, 0.0, 1.0],
                episode_summary={"success": False, "done_reason": "timeout"},
                evaluation_summary={"eval/success_rate": 0.0},
            )

            expected_names = {
                "trace": "seed_0_trace.npz",
                "xy": "seed_0_xy.png",
                "xyz": "seed_0_xyz.png",
                "summary": "summary.json",
            }
            for key, name in expected_names.items():
                path = Path(result[key])
                self.assertEqual(path.name, name)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

            with np.load(Path(tmp) / "seed_0_trace.npz") as saved:
                np.testing.assert_allclose(saved["actual_position"], actual)
                self.assertEqual(int(saved["training_step"]), 12000)
                self.assertEqual(saved["tree_xy"].shape, (1, 2))
            summary = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["training_step"], 12000)
            self.assertEqual(summary["trace_samples"], 3)

    def test_accepts_generic_static_box_metadata(self):
        trace = {
            "time_seconds": np.array([0.0, 0.1], dtype=np.float32),
            "actual_position": np.array([[-1.0, 0.0, 1.0], [-0.9, 0.0, 1.0]], dtype=np.float32),
            "actual_velocity": np.zeros((2, 3), dtype=np.float32),
            "reference_position": np.array([[-1.0, 0.0, 1.0], [-0.9, 0.0, 1.0]], dtype=np.float32),
            "reference_velocity": np.zeros((2, 3), dtype=np.float32),
            "packet_endpoint": np.array([[-0.5, 0.0, 1.0], [-0.4, 0.0, 1.0]], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            export_upper_trajectory(
                tmp,
                seed=0,
                training_step=1,
                trace=trace,
                tree_specs=[
                    {
                        "kind": "box",
                        "name": "wall",
                        "position": [0.0, 0.0, 1.4],
                        "half_extents": [0.2, 1.0, 1.4],
                    }
                ],
                start_position=[-1.0, 0.0, 1.0],
                target_position=[1.0, 0.0, 1.0],
                episode_summary={},
                evaluation_summary={},
            )
            with np.load(Path(tmp) / "seed_0_trace.npz") as saved:
                self.assertEqual(saved["obstacle_kind"].tolist(), ["box"])
                self.assertEqual(saved["obstacle_position"].shape, (1, 3))
            summary = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["obstacle_count"], 1)


if __name__ == "__main__":
    unittest.main()
