"""Pure, episode-level navigation diagnostics for evaluation.

The functions in this module operate only on sampled world positions and goal
distances.  They intentionally do not inspect environment or policy state so
the definitions are reproducible in tests and offline analysis.
"""

from __future__ import annotations

import math

import numpy as np


def episode_navigation_metrics(
    positions,
    goal_distances,
    *,
    timestep_seconds: float,
    progress_epsilon: float = 0.01,
    stuck_window_seconds: float = 1.0,
    revisit_cell_size: float = 0.25,
) -> dict[str, float]:
    """Measure trajectory efficiency, reversals, stalling, and revisits.

    ``path_efficiency`` is net goal-distance reduction divided by travelled
    distance, clipped to ``[0, 1]``.  ``backtrack_ratio`` is the fraction of
    travelled distance in transitions that increase goal distance by more than
    ``progress_epsilon``.  A no-progress span ends only after the best goal
    distance improves by at least that epsilon. ``stuck_window_fraction`` uses
    consecutive non-overlapping windows and counts windows without that net
    reduction. ``revisit_rate`` is the fraction of post-initial position
    samples that fall in an already visited 3-D spatial cell.
    """
    pos = np.asarray(positions, dtype=np.float64)
    goals = np.asarray(goal_distances, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape [samples, 3]")
    if len(pos) != len(goals) or len(pos) == 0:
        raise ValueError("positions and goal_distances must be non-empty and aligned")
    if not np.isfinite(pos).all() or not np.isfinite(goals).all():
        raise ValueError("positions and goal_distances must be finite")
    if timestep_seconds <= 0.0:
        raise ValueError("timestep_seconds must be positive")
    if progress_epsilon < 0.0 or stuck_window_seconds <= 0.0 or revisit_cell_size <= 0.0:
        raise ValueError("metric thresholds must be positive (epsilon may be zero)")

    segment_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    net_progress = max(0.0, float(goals[0] - goals[-1]))
    path_efficiency = net_progress / path_length if path_length > 0.0 else 0.0
    path_efficiency = float(np.clip(path_efficiency, 0.0, 1.0))

    goal_changes = np.diff(goals)
    backtrack_distance = float(
        np.sum(segment_lengths[goal_changes > float(progress_epsilon)])
    )
    backtrack_ratio = backtrack_distance / path_length if path_length > 0.0 else 0.0

    best_goal = float(goals[0])
    no_progress_start = None
    max_no_progress_samples = 0
    for index in range(1, len(goals)):
        if goals[index] < best_goal - progress_epsilon:
            best_goal = float(goals[index])
            if no_progress_start is not None:
                max_no_progress_samples = max(
                    max_no_progress_samples, index - no_progress_start
                )
                no_progress_start = None
        elif no_progress_start is None:
            no_progress_start = index - 1
    if no_progress_start is not None:
        max_no_progress_samples = max(
            max_no_progress_samples, len(goals) - 1 - no_progress_start
        )
    max_no_progress_seconds = float(max_no_progress_samples * timestep_seconds)

    window_steps = max(1, int(math.ceil(stuck_window_seconds / timestep_seconds)))
    stuck_windows = 0
    total_windows = 0
    for start in range(0, len(goals) - 1, window_steps):
        end = min(start + window_steps, len(goals) - 1)
        total_windows += 1
        if goals[start] - goals[end] < progress_epsilon:
            stuck_windows += 1
    stuck_window_fraction = (
        float(stuck_windows / total_windows) if total_windows else 0.0
    )

    cells = np.floor(pos / revisit_cell_size).astype(np.int64)
    visited = {tuple(cells[0])}
    revisits = 0
    for cell in cells[1:]:
        key = tuple(cell)
        if key in visited:
            revisits += 1
        else:
            visited.add(key)
    revisit_rate = float(revisits / max(1, len(cells) - 1))

    return {
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "backtrack_ratio": float(backtrack_ratio),
        "max_no_progress_seconds": max_no_progress_seconds,
        "stuck_window_fraction": stuck_window_fraction,
        "revisit_rate": revisit_rate,
    }
