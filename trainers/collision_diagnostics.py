"""Small, explicit diagnostics for the final two seconds before a collision."""

from __future__ import annotations

import math

import numpy as np


WINDOW_SECONDS = 2.0


def _vector_series(samples, key: str) -> np.ndarray:
    values = [sample.get(key, (np.nan, np.nan, np.nan)) for sample in samples]
    return np.asarray(values, dtype=np.float32).reshape(-1, 3)


def _scalar_series(samples, key: str) -> np.ndarray:
    return np.asarray([sample.get(key, np.nan) for sample in samples], dtype=np.float32)


def _sign_flips(actions: np.ndarray) -> int:
    if actions.shape[0] < 3:
        return 0
    flips = 0
    for axis in range(actions.shape[1]):
        values = actions[:, axis]
        significant = values[np.abs(values) >= 0.15]
        if significant.size >= 2:
            flips += int(np.count_nonzero(significant[1:] * significant[:-1] < 0.0))
    return flips


def classify_collision_window(samples) -> dict:
    """Classify a collision window without pretending it is a ground truth label.

    The categories deliberately match the experiment contract:
    A: no braking/evasion command; B: braking command but plant cannot settle;
    C: high-frequency command reversal; D: no range ray reported nearby geometry.
    The returned evidence makes every heuristic auditable.
    """

    if not samples:
        return {"code": "U", "label": "insufficient_window", "evidence": {}}

    command = _vector_series(samples, "cmd_velocity")
    actual = _vector_series(samples, "actual_velocity")
    action = _vector_series(samples, "upper_action")
    clearance = _scalar_series(samples, "clearance")
    sensor_fraction = _scalar_series(samples, "min_range_fraction")

    planar_command = np.linalg.norm(command[:, :2], axis=1)
    planar_actual = np.linalg.norm(actual[:, :2], axis=1)
    delta_v = np.linalg.norm(command - actual, axis=1)
    finite_clearance = clearance[np.isfinite(clearance)]
    finite_sensor = sensor_fraction[np.isfinite(sensor_fraction)]
    initial_clearance = float(finite_clearance[0]) if finite_clearance.size else math.nan
    final_clearance = float(finite_clearance[-1]) if finite_clearance.size else math.nan
    clearance_drop = (
        float(initial_clearance - final_clearance)
        if np.isfinite(initial_clearance) and np.isfinite(final_clearance)
        else math.nan
    )
    command_peak = float(np.nanmax(planar_command)) if planar_command.size else 0.0
    command_final = float(planar_command[-1]) if planar_command.size else 0.0
    actual_final = float(planar_actual[-1]) if planar_actual.size else 0.0
    delta_v_final = float(delta_v[-1]) if delta_v.size else 0.0
    action_flips = _sign_flips(action)
    min_sensor_fraction = float(np.min(finite_sensor)) if finite_sensor.size else math.nan
    closing = bool(np.isfinite(clearance_drop) and clearance_drop > 0.05)
    braking = bool(command_peak >= 0.15 and command_final <= max(0.10, 0.45 * command_peak))
    unseen = bool(np.isfinite(min_sensor_fraction) and min_sensor_fraction >= 0.98)

    evidence = {
        "initial_clearance": initial_clearance,
        "final_clearance": final_clearance,
        "clearance_drop": clearance_drop,
        "command_planar_speed_peak": command_peak,
        "command_planar_speed_final": command_final,
        "actual_planar_speed_final": actual_final,
        "delta_v_final": delta_v_final,
        "action_sign_flips": int(action_flips),
        "min_range_fraction": min_sensor_fraction,
        "closing": closing,
        "braking_command": braking,
    }
    if unseen:
        return {"code": "D", "label": "no_ray_warning", "evidence": evidence}
    if closing and braking and actual_final > command_final + 0.12 and delta_v_final > 0.12:
        return {"code": "B", "label": "brake_command_not_realized", "evidence": evidence}
    if closing and action_flips >= 3:
        return {"code": "C", "label": "oscillatory_local_command", "evidence": evidence}
    if closing and command_final >= 0.75 * max(command_peak, 1e-6):
        return {"code": "A", "label": "no_brake_or_evasion_command", "evidence": evidence}
    return {"code": "U", "label": "mixed_or_insufficient_evidence", "evidence": evidence}

