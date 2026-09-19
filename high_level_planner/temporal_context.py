"""Explicit short-term environmental context for the reference planner.

The frozen 16-D latent already receives the legacy motor-action history.  The
features here deliberately do *not* repeat that 240-D action history: they
describe how goal, sparse range sensing, velocity, and yaw evolve across
successive high-level decisions.
"""

from __future__ import annotations

import numpy as np

from envs.observation_layout import ForestObservationLayout


HISTORY_LENGTH = 8
TEMPORAL_FRAME_DIM = 16


def extract_environment_temporal_frame(
    preprocessed_observation,
    *,
    action_dim: int = 4,
) -> np.ndarray:
    """Return one 16-D environment/motion frame.

    Layout: normalized relative goal (3), world-fixed eight-ray ranges (8),
    normalized linear velocity (3), and yaw represented as sin/cos (2).
    Goal/range channels intentionally retain the existing planner's coordinate
    contract.  The yaw pair makes their temporal evolution interpretable
    without changing the sensor or frozen Encoder.
    """

    observation = np.asarray(preprocessed_observation, dtype=np.float32).reshape(-1)
    layout = ForestObservationLayout.from_total_dim(observation.size, action_dim=int(action_dim))
    parts = layout.split(observation)
    kin = np.asarray(parts.kin, dtype=np.float32)
    if kin.shape != (12,):
        raise ValueError(f"Expected 12-D kinematic observation, got {kin.shape}.")

    # preprocess_state normalizes yaw by pi and linear velocity by 2.0.  Using
    # the normalized velocity retains the legacy observation scale, while
    # sin/cos avoids a discontinuity at +/-pi.
    yaw = float(kin[5]) * float(np.pi)
    frame = np.concatenate(
        [
            np.asarray(parts.goal, dtype=np.float32),
            np.asarray(parts.ranges, dtype=np.float32),
            kin[6:9],
            np.asarray([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
        ]
    ).astype(np.float32)
    if frame.shape != (TEMPORAL_FRAME_DIM,) or not np.all(np.isfinite(frame)):
        raise ValueError("Temporal environment frame must be finite 16-D values.")
    return frame


def append_temporal_frame(history: list[np.ndarray], frame) -> tuple[np.ndarray, int]:
    """Append one high-level frame and return fixed 8-frame repeat padding.

    At an episode's first decision the only available frame is repeated.  The
    returned padding count is stored for diagnostics, because step-zero errors
    cannot be credited to a history signal that did not yet exist.
    """

    value = np.asarray(frame, dtype=np.float32).reshape(TEMPORAL_FRAME_DIM).copy()
    history.append(value)
    if len(history) > HISTORY_LENGTH:
        del history[:-HISTORY_LENGTH]
    padding = max(0, HISTORY_LENGTH - len(history))
    sequence = [history[0]] * padding + history
    result = np.asarray(sequence, dtype=np.float32)
    if result.shape != (HISTORY_LENGTH, TEMPORAL_FRAME_DIM):
        raise AssertionError(f"Unexpected temporal history shape {result.shape}.")
    return result, padding
