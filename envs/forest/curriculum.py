from __future__ import annotations

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
    ):
        self.enabled = bool(enabled)
        self.milestones = tuple(int(x) for x in milestones)
        if len(self.milestones) != 3:
            raise ValueError("curriculum_milestones must contain exactly 3 episode counts")

        self.corridor_half_width = float(corridor_half_width)
        self.wide_corridor_half_width = float(max(wide_corridor_half_width, corridor_half_width))
        self.narrow_corridor_half_width = float(min(narrow_corridor_half_width, corridor_half_width))
        self.centerline_tree_fraction = float(centerline_tree_fraction)

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
        stage = self.stage_from_episode_count(completed_episodes) if override_stage is None else int(override_stage)
        return int(stage), self.stage_config(int(stage))
