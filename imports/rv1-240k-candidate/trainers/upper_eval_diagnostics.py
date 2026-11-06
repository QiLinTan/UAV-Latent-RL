"""Deterministic, evaluation-only probes for the learned upper controller."""

from __future__ import annotations

import numpy as np


Z_ACTION_SCAN_VALUES = np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0), dtype=np.float32)


def p_height_ablation_action(
    policy_action,
    *,
    position,
    target_position,
    max_vertical_speed: float,
    gain: float,
) -> tuple[np.ndarray, float]:
    """Keep the policy xy action and replace only its z command with P control."""

    action = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
    if action.size < 3:
        raise ValueError("P-height ablation requires a 3-D upper action.")
    if max_vertical_speed <= 0.0 or gain <= 0.0:
        raise ValueError("max_vertical_speed and gain must be positive.")

    position = np.asarray(position, dtype=np.float32).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    command_z = float(
        np.clip(
            float(gain) * float(target_position[2] - position[2]),
            -float(max_vertical_speed),
            float(max_vertical_speed),
        )
    )
    action[2] = command_z / float(max_vertical_speed)
    return np.clip(action, -1.0, 1.0), command_z


def z_action_candidates(policy_action, scan_values=Z_ACTION_SCAN_VALUES) -> np.ndarray:
    """Return actions with fixed xy and a sweep over normalized z actions."""

    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.size < 3:
        raise ValueError("Z-action scan requires a 3-D upper action.")
    values = np.asarray(scan_values, dtype=np.float32).reshape(-1)
    if values.size == 0 or np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("scan_values must be a non-empty subset of [-1, 1].")
    candidates = np.repeat(action.reshape(1, -1), values.size, axis=0)
    candidates[:, 2] = values
    return candidates
