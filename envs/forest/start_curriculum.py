"""Training-only reverse start curriculum for the procedural forest."""

from collections import deque

import numpy as np

from benchmarks.forest_benchmark import grid_route


START_DISTANCE_BANDS = ((0.35, 1.2), (1.2, 2.5), (2.5, 4.5), (4.5, 7.0))
ORIGINAL_START_PROBABILITIES = (0.20, 0.20, 0.30, 0.50, 1.0)


class StartDistanceCurriculum:
    def __init__(self, *, window=100, minimum_episodes=100,
                 success_threshold=0.70, max_collision_rate=0.20,
                 bands=START_DISTANCE_BANDS, original_probabilities=ORIGINAL_START_PROBABILITIES):
        if window <= 0 or minimum_episodes <= 0:
            raise ValueError("Start curriculum window and minimum episodes must be positive")
        self.window = int(window)
        self.minimum_episodes = int(minimum_episodes)
        self.success_threshold = float(success_threshold)
        self.max_collision_rate = float(max_collision_rate)
        if not 0 <= self.success_threshold <= 1 or not 0 <= self.max_collision_rate <= 1:
            raise ValueError("Start curriculum rates must lie in [0, 1]")
        self.bands = tuple(tuple(band) for band in bands)
        self.original_probabilities = tuple(original_probabilities)
        if len(self.original_probabilities) != len(self.bands) + 1 or any(
                not 0 <= p <= 1 for p in self.original_probabilities):
            raise ValueError("Invalid start curriculum probabilities")
        if any(not 0 < low < high for low, high in self.bands):
            raise ValueError("Invalid start curriculum bands")
        self.level = 0
        self.level_episodes = 0
        self.successes = deque(maxlen=self.window)
        self.collisions = deque(maxlen=self.window)

    @property
    def original_probability(self):
        return self.original_probabilities[self.level]

    @property
    def complete(self):
        return self.level >= len(self.bands)

    @property
    def success_rate(self):
        return float(np.mean(self.successes)) if self.successes else 0.0

    @property
    def collision_rate(self):
        return float(np.mean(self.collisions)) if self.collisions else 0.0

    def record(self, *, source, success, collision):
        # Full-distance mixing and planner fallbacks cannot advance a short band.
        if self.level >= len(self.bands) or source != "band":
            return
        self.level_episodes += 1
        self.successes.append(bool(success))
        self.collisions.append(bool(collision))
        if (self.level_episodes >= self.minimum_episodes
                and len(self.successes) >= min(self.window, self.minimum_episodes)
                and self.success_rate >= self.success_threshold
                and self.collision_rate <= self.max_collision_rate):
            self.level += 1
            self.level_episodes = 0
            self.successes.clear()
            self.collisions.clear()

    def metrics(self):
        return {
            "start_curriculum_level": self.level,
            "start_curriculum_level_episodes": self.level_episodes,
            "start_curriculum_window_episodes": len(self.successes),
            "start_curriculum_success_rate": self.success_rate,
            "start_curriculum_collision_rate": self.collision_rate,
            "start_curriculum_original_probability": self.original_probability,
        }


def path_start_candidates(base_env, band):
    """Return collision-margin path nodes within a remaining-path-length band.

    The privileged path is used only to choose reset positions. It is not an
    observation, action, reward, or replay feature.
    """
    record = {
        "start": np.asarray(base_env.START_POS),
        "goal": np.asarray(base_env.TARGET_POS),
        "trees": base_env._tree_specs,
    }
    if not record["trees"] or any("xy" not in tree for tree in record["trees"]):
        return np.empty((0, 3), dtype=np.float32)
    path = grid_route(record, resolution=0.1,
                      inflation=float(base_env.COLLISION_R) + 0.22,
                      bound=max(float(base_env.FOREST_HALF_EXTENT), 4.0))
    if path is None:
        return np.empty((0, 3), dtype=np.float32)
    remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)[::-1])[::-1], 0.0]
    points = []
    for xy, length in zip(path, remaining):
        if not band[0] <= length < band[1]:
            continue
        pos = np.array([xy[0], xy[1], base_env.TARGET_POS[2]], dtype=np.float32)
        if np.linalg.norm(pos - base_env.TARGET_POS) <= max(0.30, base_env.GOAL_TOLERANCE + 0.10):
            continue
        if np.any(np.abs(pos[:2]) >= base_env.FOREST_HALF_EXTENT - 0.25):
            continue
        if base_env._computeNearestTreeClearance(pos) < 0.18:
            continue
        points.append(pos)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)
