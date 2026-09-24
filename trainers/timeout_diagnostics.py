"""Auditable timeout failure classification for upper-navigation evaluation."""

from __future__ import annotations

import math

import numpy as np


def _vectors(samples, key: str) -> np.ndarray:
    return np.asarray(
        [sample.get(key, (np.nan, np.nan, np.nan)) for sample in samples],
        dtype=np.float32,
    ).reshape(-1, 3)


def classify_timeout(
    *,
    positions,
    goal_distances,
    target_position,
    navigation_metrics,
    last_window_samples,
    final_region_radius: float = 1.2,
    near_goal_radius: float = 0.8,
    height_tolerance: float = 0.25,
) -> dict:
    """Classify a timeout as T1–T4 using only logged episode evidence.

    T1: entered the near-goal region but loitered or departed.
    T2: never reached the final region.
    T3: xy approached the target but terminal height did not converge.
    T4: command and actual motion became excessively conservative.
    """

    pos = np.asarray(positions, dtype=np.float32)
    goals = np.asarray(goal_distances, dtype=np.float32).reshape(-1)
    target = np.asarray(target_position, dtype=np.float32).reshape(3)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) != len(goals) or len(pos) == 0:
        raise ValueError("positions and goal_distances must be aligned non-empty [N, 3]/[N].")

    xy_errors = np.linalg.norm(pos[:, :2] - target[:2], axis=1)
    min_goal_distance = float(np.min(goals))
    final_goal_distance = float(goals[-1])
    min_xy_error = float(np.min(xy_errors))
    final_xy_error = float(xy_errors[-1])
    final_z_error = float(target[2] - pos[-1, 2])

    command = _vectors(last_window_samples, "cmd_velocity")
    actual = _vectors(last_window_samples, "actual_velocity")
    clearance = np.asarray(
        [sample.get("clearance", np.nan) for sample in last_window_samples],
        dtype=np.float32,
    )
    command_planar = np.linalg.norm(command[:, :2], axis=1)
    actual_planar = np.linalg.norm(actual[:, :2], axis=1)
    finite_clearance = clearance[np.isfinite(clearance)]
    evidence = {
        "min_goal_distance": min_goal_distance,
        "final_goal_distance": final_goal_distance,
        "min_xy_error": min_xy_error,
        "final_xy_error": final_xy_error,
        "final_z_error": final_z_error,
        "backtrack_ratio": float(navigation_metrics.get("backtrack_ratio", math.nan)),
        "last_window_cmd_velocity_mean": np.nanmean(command, axis=0).tolist(),
        "last_window_actual_velocity_mean": np.nanmean(actual, axis=0).tolist(),
        "last_window_command_planar_speed_mean": float(np.nanmean(command_planar)),
        "last_window_actual_planar_speed_mean": float(np.nanmean(actual_planar)),
        "last_window_clearance_min": (
            float(np.min(finite_clearance)) if finite_clearance.size else math.nan
        ),
        "last_window_samples": int(len(last_window_samples)),
    }

    conservative = bool(
        evidence["last_window_command_planar_speed_mean"] < 0.12
        and evidence["last_window_actual_planar_speed_mean"] < 0.12
        and min_goal_distance > near_goal_radius
    )
    if min_xy_error <= final_region_radius and abs(final_z_error) > height_tolerance:
        return {"code": "T3", "label": "xy_close_height_not_converged", "evidence": evidence}
    if conservative:
        return {"code": "T4", "label": "overly_conservative_low_motion", "evidence": evidence}
    if min_goal_distance <= near_goal_radius:
        return {"code": "T1", "label": "near_goal_loiter_or_departure", "evidence": evidence}
    if min_goal_distance > final_region_radius:
        return {"code": "T2", "label": "never_reached_final_region", "evidence": evidence}
    return {"code": "TU", "label": "mixed_terminal_timeout", "evidence": evidence}
