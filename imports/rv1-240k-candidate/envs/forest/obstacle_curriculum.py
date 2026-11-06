"""Obstacle-stage scheduler gated by full-route, primary-difficulty episodes."""

from collections import deque


class ObstacleCurriculum:
    def __init__(self, *, route_counts=(1, 2, 3, 5), window=100,
                 minimum_episodes=100, success_thresholds=(0.80, 0.75, 0.70),
                 max_collision_rates=(0.15, 0.20, 0.20)):
        self.route_counts = tuple(int(value) for value in route_counts)
        self.success_thresholds = tuple(float(value) for value in success_thresholds)
        self.max_collision_rates = tuple(float(value) for value in max_collision_rates)
        self.window = int(window)
        self.minimum_episodes = int(minimum_episodes)
        if len(self.route_counts) != 4 or len(self.success_thresholds) != 3 \
                or len(self.max_collision_rates) != 3:
            raise ValueError("Obstacle curriculum requires four stages and three gates")
        if self.window <= 0 or self.minimum_episodes <= 0:
            raise ValueError("Obstacle curriculum window and minimum episodes must be positive")
        if any(not 0.0 <= value <= 1.0 for value in
               self.success_thresholds + self.max_collision_rates):
            raise ValueError("Obstacle curriculum rates must lie in [0, 1]")
        self.stage = 0
        self.stage_episodes = 0
        self.successes = deque(maxlen=self.window)
        self.collisions = deque(maxlen=self.window)

    @property
    def success_rate(self):
        return sum(self.successes) / len(self.successes) if self.successes else 0.0

    @property
    def collision_rate(self):
        return sum(self.collisions) / len(self.collisions) if self.collisions else 0.0

    def record(self, *, episode_stage, start_level, primary_difficulty,
               success, collision, start_complete=None):
        """Return True exactly when the current stage advances.

        Short-start and easier mixed episodes may supply replay but cannot
        certify full-route navigation at the stage's primary obstacle count.
        """
        if start_complete is None:
            start_complete = start_level == 4
        if episode_stage != self.stage or not start_complete or not primary_difficulty:
            return False
        self.stage_episodes += 1
        self.successes.append(bool(success))
        self.collisions.append(bool(collision))
        if self.stage >= 3:
            return False
        if (self.stage_episodes < self.minimum_episodes
                or len(self.successes) < min(self.window, self.minimum_episodes)
                or self.success_rate < self.success_thresholds[self.stage]
                or self.collision_rate > self.max_collision_rates[self.stage]):
            return False
        self.stage += 1
        self.stage_episodes = 0
        self.successes.clear()
        self.collisions.clear()
        return True

    def metrics(self):
        return {
            "obstacle_curriculum_stage": self.stage,
            "obstacle_curriculum_primary_route_count": self.route_counts[self.stage],
            "obstacle_curriculum_stage_episodes": self.stage_episodes,
            "obstacle_curriculum_window_episodes": len(self.successes),
            "obstacle_curriculum_success_rate": self.success_rate,
            "obstacle_curriculum_collision_rate": self.collision_rate,
            "obstacle_curriculum_success_threshold": (
                self.success_thresholds[self.stage] if self.stage < 3 else None
            ),
            "obstacle_curriculum_max_collision_rate": (
                self.max_collision_rates[self.stage] if self.stage < 3 else None
            ),
        }
