from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


KIN_DIM = 12
GOAL_DIM = 3
RANGE_DIM = 8
FOREST_TASK_DIM = GOAL_DIM + RANGE_DIM


class ForestObservationParts(NamedTuple):
    kin: object
    action_history: object
    goal: object
    ranges: object


@dataclass(frozen=True)
class ForestObservationLayout:
    """Slices for KIN + action history + goal + range observations."""

    total_dim: int
    action_history_dim: int

    @classmethod
    def from_total_dim(cls, total_dim: int, action_dim: int | None = None):
        total_dim = int(total_dim)
        action_history_dim = total_dim - KIN_DIM - FOREST_TASK_DIM
        if action_history_dim < 0:
            raise ValueError(
                f"Forest observation needs at least {KIN_DIM + FOREST_TASK_DIM} values, got {total_dim}."
            )
        if action_dim is not None and action_history_dim % int(action_dim) != 0:
            raise ValueError(
                f"Action-history dimension {action_history_dim} is not divisible by action dimension {action_dim}."
            )
        return cls(total_dim=total_dim, action_history_dim=action_history_dim)

    @property
    def goal_start(self) -> int:
        return KIN_DIM + self.action_history_dim

    @property
    def range_start(self) -> int:
        return self.goal_start + GOAL_DIM

    def split(self, observation) -> ForestObservationParts:
        actual_dim = int(observation.shape[-1])
        if actual_dim != self.total_dim:
            raise ValueError(f"Expected observation dimension {self.total_dim}, got {actual_dim}.")
        return ForestObservationParts(
            kin=observation[..., :KIN_DIM],
            action_history=observation[..., KIN_DIM : self.goal_start],
            goal=observation[..., self.goal_start : self.range_start],
            ranges=observation[..., self.range_start : self.total_dim],
        )


def append_forest_task_observation(base_observation, goal_observation, range_observation):
    """Append fixed-size forest task observations to a base KIN observation."""

    base = np.asarray(base_observation)
    goal = np.asarray(goal_observation)
    ranges = np.asarray(range_observation)
    if goal.shape[-1] != GOAL_DIM:
        raise ValueError(f"Expected {GOAL_DIM} goal values, got {goal.shape[-1]}.")
    if ranges.shape[-1] != RANGE_DIM:
        raise ValueError(f"Expected {RANGE_DIM} range values, got {ranges.shape[-1]}.")
    if base.shape[:-1] != goal.shape[:-1] or base.shape[:-1] != ranges.shape[:-1]:
        raise ValueError(
            "Base, goal, and range observations must have identical leading dimensions."
        )
    return np.concatenate([base, goal, ranges], axis=-1)
