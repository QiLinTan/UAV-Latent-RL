"""Diagnostics for near-goal guidance and successful terminal transitions."""

from __future__ import annotations

import json
import os

import numpy as np


DEFAULT_DISTANCE_BINS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _optional_mean(values: np.ndarray):
    return float(np.mean(values)) if values.size else None


def summarize_terminal_episode(
    *,
    positions,
    velocities,
    target_position,
    goal_tolerance: float,
    guidance_radius: float,
    done_reason: str,
    timestep_seconds: float,
    distance_bins=DEFAULT_DISTANCE_BINS,
) -> dict:
    """Summarize terminal-region coverage for one complete evaluation rollout.

    ``positions`` and ``velocities`` include the reset state followed by one state
    for each environment transition. Coverage counts are therefore based on the
    post-transition states at indices 1..N.
    """

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3].")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions.")
    if positions.shape[0] < 2:
        raise ValueError("At least an initial and one post-transition state are required.")
    if not 0.0 < goal_tolerance < guidance_radius:
        raise ValueError("Require 0 < goal_tolerance < guidance_radius.")
    if timestep_seconds <= 0.0:
        raise ValueError("timestep_seconds must be positive.")

    goal_error = target[None, :] - positions
    distance_3d = np.linalg.norm(goal_error, axis=1)
    distance_xy = np.linalg.norm(goal_error[:, :2], axis=1)
    abs_z_error = np.abs(goal_error[:, 2])

    previous_distance = distance_3d[:-1]
    distance = distance_3d[1:]
    post_error = goal_error[1:]
    post_velocity = velocities[1:]
    guidance_mask = (distance >= goal_tolerance) & (distance < guidance_radius)
    success_mask = distance < goal_tolerance
    entry_mask = (previous_distance >= guidance_radius) & (distance < guidance_radius)
    success_crossing_mask = (previous_distance >= goal_tolerance) & success_mask

    safe_distance = np.maximum(distance, 1e-9)
    goal_direction = post_error / safe_distance[:, None]
    closing_speed = np.sum(post_velocity * goal_direction, axis=1)
    tangent_velocity = post_velocity - closing_speed[:, None] * goal_direction
    tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
    planar_speed = np.linalg.norm(post_velocity[:, :2], axis=1)

    bin_edges = np.asarray(distance_bins, dtype=np.float64)
    if (
        bin_edges.ndim != 1
        or bin_edges.size < 2
        or np.any(np.diff(bin_edges) <= 0.0)
    ):
        raise ValueError("distance_bins must be a strictly increasing 1-D sequence.")
    distance_bin_counts = {}
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        label = f"[{lower:.1f},{upper:.1f})"
        distance_bin_counts[label] = int(np.sum((distance >= lower) & (distance < upper)))

    guidance_closing = closing_speed[guidance_mask]
    guidance_tangent = tangent_speed[guidance_mask]
    guidance_planar = planar_speed[guidance_mask]
    xy_close_3d_outside = (
        (distance_xy[1:] < guidance_radius) & (distance >= guidance_radius)
    )

    min_index = int(np.argmin(distance_3d))
    return {
        "done_reason": str(done_reason),
        "transition_count": int(distance.size),
        "goal_tolerance_m": float(goal_tolerance),
        "guidance_radius_m": float(guidance_radius),
        "entered_guidance_region": bool(
            np.any(entry_mask)
            or (goal_tolerance <= distance_3d[0] < guidance_radius)
        ),
        "guidance_entry_count": int(np.sum(entry_mask)),
        "guidance_state_count": int(np.sum(guidance_mask)),
        "guidance_time_seconds": float(np.sum(guidance_mask) * timestep_seconds),
        "guidance_fraction": float(np.mean(guidance_mask)),
        "longest_guidance_run_steps": _longest_true_run(guidance_mask),
        "longest_guidance_run_seconds": float(
            _longest_true_run(guidance_mask) * timestep_seconds
        ),
        "successful_terminal_transition_count": int(np.sum(success_crossing_mask)),
        "reached_success_ball": bool(np.any(success_mask)),
        "minimum_goal_distance_m": float(distance_3d[min_index]),
        "xy_error_at_minimum_goal_distance_m": float(distance_xy[min_index]),
        "abs_z_error_at_minimum_goal_distance_m": float(abs_z_error[min_index]),
        "minimum_xy_error_m": float(np.min(distance_xy)),
        "minimum_abs_z_error_m": float(np.min(abs_z_error)),
        "xy_close_but_3d_outside_state_count": int(np.sum(xy_close_3d_outside)),
        "distance_bin_state_counts": distance_bin_counts,
        "guidance_mean_closing_speed_mps": _optional_mean(guidance_closing),
        "guidance_away_motion_fraction": (
            float(np.mean(guidance_closing < 0.0)) if guidance_closing.size else None
        ),
        "guidance_mean_tangential_speed_mps": _optional_mean(guidance_tangent),
        "guidance_mean_planar_speed_mps": _optional_mean(guidance_planar),
    }


def summarize_terminal_coverage(episodes: list[dict]) -> dict:
    if not episodes:
        return {"episode_count": 0}

    entered = [item for item in episodes if item["entered_guidance_region"]]
    successful = [
        item for item in episodes if item["successful_terminal_transition_count"] > 0
    ]
    total_transitions = sum(int(item["transition_count"]) for item in episodes)
    total_guidance_states = sum(int(item["guidance_state_count"]) for item in episodes)
    bin_names = list(episodes[0]["distance_bin_state_counts"])
    aggregate_bins = {
        name: int(sum(item["distance_bin_state_counts"].get(name, 0) for item in episodes))
        for name in bin_names
    }
    return {
        "episode_count": int(len(episodes)),
        "guidance_entered_episode_count": int(len(entered)),
        "guidance_entered_episode_fraction": float(len(entered) / len(episodes)),
        "guidance_state_count": int(total_guidance_states),
        "guidance_state_fraction": (
            float(total_guidance_states / total_transitions) if total_transitions else 0.0
        ),
        "successful_terminal_transition_count": int(
            sum(item["successful_terminal_transition_count"] for item in episodes)
        ),
        "successful_episode_count": int(len(successful)),
        "minimum_goal_distance_m": float(
            min(item["minimum_goal_distance_m"] for item in episodes)
        ),
        "xy_close_but_3d_outside_state_count": int(
            sum(item["xy_close_but_3d_outside_state_count"] for item in episodes)
        ),
        "distance_bin_state_counts": aggregate_bins,
    }


def export_terminal_coverage(
    output_dir,
    *,
    episodes: list[dict],
    training_step: int,
    timestep_seconds: float,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": "upper_terminal_coverage_v1",
        "training_step": int(training_step),
        "decision_rate_hz": float(1.0 / timestep_seconds),
        "summary": summarize_terminal_coverage(episodes),
        "episodes": episodes,
    }
    latest_path = os.path.join(output_dir, "terminal_coverage_latest.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    history_path = os.path.join(output_dir, "terminal_coverage.jsonl")
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"json": latest_path, "jsonl": history_path, "summary": payload["summary"]}
