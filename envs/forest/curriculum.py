from __future__ import annotations

from collections import deque
from dataclasses import dataclass


DEFAULT_CURRICULUM_MILESTONES = (800, 2500, 6000)


@dataclass(frozen=True)
class ForestCurriculumStage:
    corridor_half_width: float
    protect_corridor: bool
    corridor_edge_tree_fraction: float
    centerline_tree_fraction: float


class ForestCurriculumScheduler:
    def __init__(
        self,
        *,
        enabled: bool = True,
        milestones=DEFAULT_CURRICULUM_MILESTONES,
        corridor_half_width: float,
        wide_corridor_half_width: float,
        narrow_corridor_half_width: float,
        centerline_tree_fraction: float,
        success_gated: bool = False,
        success_window: int = 100,
        success_thresholds=(0.10, 0.20, 0.30),
        minimum_stage_episodes: int = 50,
    ):
        self.enabled = bool(enabled)
        self.milestones = tuple(int(x) for x in milestones)
        if len(self.milestones) != 3:
            raise ValueError("curriculum_milestones must contain exactly 3 episode counts")

        self.corridor_half_width = float(corridor_half_width)
        self.wide_corridor_half_width = float(max(wide_corridor_half_width, corridor_half_width))
        self.narrow_corridor_half_width = float(min(narrow_corridor_half_width, corridor_half_width))
        self.centerline_tree_fraction = float(centerline_tree_fraction)
        self.success_gated = bool(success_gated)
        self.success_window = int(success_window)
        self.success_thresholds = tuple(float(x) for x in success_thresholds)
        self.minimum_stage_episodes = int(minimum_stage_episodes)
        if len(self.success_thresholds) != 3:
            raise ValueError("success_thresholds must contain exactly 3 values")
        if self.success_window <= 0 or self.minimum_stage_episodes <= 0:
            raise ValueError("success_window and minimum_stage_episodes must be positive")
        self._performance_stage = 0
        self._stage_episode_count = 0
        self._recent_successes = deque(maxlen=self.success_window)

    def record_episode_outcome(self, success: bool):
        if not self.enabled or not self.success_gated:
            return
        self._recent_successes.append(float(bool(success)))
        self._stage_episode_count += 1
        if self._performance_stage >= 3:
            return
        enough_samples = (
            self._stage_episode_count >= self.minimum_stage_episodes
            and len(self._recent_successes) >= min(self.success_window, self.minimum_stage_episodes)
        )
        if enough_samples and self.success_rate >= self.success_thresholds[self._performance_stage]:
            self._performance_stage += 1
            self._stage_episode_count = 0
            self._recent_successes.clear()

    @property
    def success_rate(self) -> float:
        if not self._recent_successes:
            return 0.0
        return float(sum(self._recent_successes) / len(self._recent_successes))

    @property
    def recent_episode_count(self) -> int:
        return len(self._recent_successes)

    def stage_from_episode_count(self, completed_episodes: int) -> int:
        first, second, third = self.milestones
        if not self.enabled:
            return 3
        if completed_episodes < first:
            return 0
        if completed_episodes < second:
            return 1
        if completed_episodes < third:
            return 2
        return 3

    def stage_config(self, stage: int) -> ForestCurriculumStage:
        medium_corridor = max(self.corridor_half_width, 0.95)
        protected_narrow_corridor = max(self.narrow_corridor_half_width, 0.65)
        unprotected_narrow_corridor = max(self.narrow_corridor_half_width, 0.35)

        if stage <= 0:
            return ForestCurriculumStage(
                corridor_half_width=self.wide_corridor_half_width,
                protect_corridor=True,
                corridor_edge_tree_fraction=0.0,
                centerline_tree_fraction=0.0,
            )
        if stage == 1:
            return ForestCurriculumStage(
                corridor_half_width=medium_corridor,
                protect_corridor=True,
                corridor_edge_tree_fraction=0.0,
                centerline_tree_fraction=0.0,
            )
        if stage == 2:
            return ForestCurriculumStage(
                corridor_half_width=protected_narrow_corridor,
                protect_corridor=True,
                corridor_edge_tree_fraction=min(0.15, self.centerline_tree_fraction),
                centerline_tree_fraction=0.0,
            )
        return ForestCurriculumStage(
            corridor_half_width=unprotected_narrow_corridor,
            protect_corridor=False,
            corridor_edge_tree_fraction=0.0,
            centerline_tree_fraction=min(0.15, self.centerline_tree_fraction),
        )

    def resolve(self, completed_episodes: int, override_stage: int | None = None) -> tuple[int, ForestCurriculumStage]:
        if override_stage is not None:
            stage = int(override_stage)
        elif not self.enabled:
            stage = 3
        elif self.success_gated:
            stage = self._performance_stage
        else:
            stage = self.stage_from_episode_count(completed_episodes)
        return int(stage), self.stage_config(int(stage))
