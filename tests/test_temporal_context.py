from __future__ import annotations

import unittest

import numpy as np
import torch

from high_level_planner import (
    HISTORY_LENGTH,
    TEMPORAL_FRAME_DIM,
    TemporalReferenceSequencePlanner,
    append_temporal_frame,
    extract_environment_temporal_frame,
)


class TemporalContextTest(unittest.TestCase):
    def test_frame_layout_uses_goal_ranges_velocity_and_yaw(self):
        state = np.zeros(263, dtype=np.float32)
        state[5] = 0.5  # normalized yaw = pi/2
        state[6:9] = (0.1, 0.2, 0.3)
        state[252:255] = (0.4, -0.5, 0.6)
        state[255:263] = np.arange(8, dtype=np.float32) / 10.0
        frame = extract_environment_temporal_frame(state)
        self.assertEqual(frame.shape, (TEMPORAL_FRAME_DIM,))
        np.testing.assert_allclose(frame[:3], state[252:255])
        np.testing.assert_allclose(frame[3:11], state[255:263])
        np.testing.assert_allclose(frame[11:14], state[6:9])
        np.testing.assert_allclose(frame[14:], (1.0, 0.0), atol=1e-6)

    def test_first_frame_repeat_padding_then_history_advances(self):
        history = []
        first, padding = append_temporal_frame(history, np.zeros(TEMPORAL_FRAME_DIM, dtype=np.float32))
        self.assertEqual(padding, HISTORY_LENGTH - 1)
        np.testing.assert_allclose(first, 0.0)
        second, padding = append_temporal_frame(history, np.ones(TEMPORAL_FRAME_DIM, dtype=np.float32))
        self.assertEqual(padding, HISTORY_LENGTH - 2)
        np.testing.assert_allclose(second[:-1], 0.0)
        np.testing.assert_allclose(second[-1], 1.0)

    def test_variants_preserve_30d_reference_contract(self):
        current = torch.zeros(2, 27)
        history = torch.zeros(2, HISTORY_LENGTH, TEMPORAL_FRAME_DIM)
        for variant in ("history_stack", "gru"):
            model = TemporalReferenceSequencePlanner(variant)
            output = model(current, history)
            self.assertEqual(tuple(output.shape), (2, 30))
            self.assertEqual(tuple(model.denormalize(output).shape), (2, 5, 6))


if __name__ == "__main__":
    unittest.main()
