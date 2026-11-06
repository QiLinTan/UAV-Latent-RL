import unittest

import numpy as np

from trainers.upper_eval_diagnostics import p_height_ablation_action, z_action_candidates


class UpperEvalDiagnosticsTest(unittest.TestCase):
    def test_p_height_ablation_preserves_xy_and_drives_toward_target_height(self):
        action, command_z = p_height_ablation_action(
            np.array([0.4, -0.3, 0.8], dtype=np.float32),
            position=np.array([0.0, 0.0, 1.5], dtype=np.float32),
            target_position=np.array([1.0, 2.0, 1.0], dtype=np.float32),
            max_vertical_speed=0.3,
            gain=1.0,
        )
        np.testing.assert_allclose(action[:2], [0.4, -0.3])
        self.assertAlmostEqual(command_z, -0.3)
        self.assertAlmostEqual(float(action[2]), -1.0)

    def test_z_action_candidates_preserve_xy(self):
        candidates = z_action_candidates(np.array([0.2, -0.4, 0.7], dtype=np.float32))
        self.assertEqual(candidates.shape, (5, 3))
        np.testing.assert_allclose(candidates[:, :2], [[0.2, -0.4]] * 5)
        np.testing.assert_allclose(candidates[:, 2], [-1.0, -0.5, 0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
