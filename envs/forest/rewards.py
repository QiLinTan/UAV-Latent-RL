from __future__ import annotations

import numpy as np

from .geometry import distance_point_to_line_2d, route_projection_2d


def default_reward_terms():
    return {
        "progress_reward": 0.0,
        "distance_reward": 0.0,
        "goal_bonus": 0.0,
        "height_penalty": 0.0,
        "vertical_speed_penalty": 0.0,
        "lateral_penalty": 0.0,
        "proximity_penalty": 0.0,
        "attitude_penalty": 0.0,
        "speed_penalty": 0.0,
        "collision_penalty": 0.0,
        "route_lateral_offset": 0.0,
        "goal_progress": 0.0,
        "route_progress": 0.0,
    }


class ForestRewardModel:
    def compute(
        self,
        *,
        state,
        prev_goal_dist: float | None,
        prev_pos,
        start_pos,
        target_pos,
        goal_tolerance: float,
        safe_distance: float,
        clearance: float,
        collision: bool,
    ):
        raise NotImplementedError


class BaselineForestReward(ForestRewardModel):
    def compute(
        self,
        *,
        state,
        prev_goal_dist: float | None,
        prev_pos,
        start_pos,
        target_pos,
        goal_tolerance: float,
        safe_distance: float,
        clearance: float,
        collision: bool,
    ):
        state = np.asarray(state, dtype=np.float32)
        pos = state[0:3]
        roll, pitch = state[7], state[8]
        vel = state[10:13]

        start_pos = np.asarray(start_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        prev_pos = np.asarray(prev_pos, dtype=np.float32)

        goal_dist = float(np.linalg.norm(target_pos - pos))
        reference_goal_dist = goal_dist if prev_goal_dist is None else float(prev_goal_dist)
        goal_progress = reference_goal_dist - goal_dist
        route_progress = route_projection_2d(pos[:2], start_pos[:2], target_pos[:2]) - route_projection_2d(
            prev_pos[:2], start_pos[:2], target_pos[:2]
        )

        progress_reward = 35.0 * route_progress + 15.0 * goal_progress
        distance_reward = 0.5 / (1.0 + goal_dist)
        goal_bonus = 120.0 if goal_dist < goal_tolerance else 0.0

        target_z = float(target_pos[2])
        route_lateral_offset = distance_point_to_line_2d(pos[:2], start_pos[:2], target_pos[:2])
        height_penalty = 0.8 * abs(float(pos[2]) - target_z)
        vertical_speed_penalty = 0.12 * abs(float(vel[2]))
        lateral_penalty = 0.15 * route_lateral_offset

        proximity_penalty = 0.0
        if clearance < safe_distance:
            proximity_penalty = 2.5 * ((safe_distance - clearance) / max(safe_distance, 1e-6)) ** 2

        attitude_penalty = 0.10 * (abs(roll) + abs(pitch))
        speed_penalty = 0.003 * float(np.linalg.norm(vel))
        collision_penalty = 25.0 if collision else 0.0

        reward = progress_reward + distance_reward + goal_bonus
        reward -= height_penalty + vertical_speed_penalty + lateral_penalty
        reward -= proximity_penalty + attitude_penalty + speed_penalty + collision_penalty

        reward_terms = {
            "progress_reward": float(progress_reward),
            "distance_reward": float(distance_reward),
            "goal_bonus": float(goal_bonus),
            "height_penalty": float(height_penalty),
            "vertical_speed_penalty": float(vertical_speed_penalty),
            "lateral_penalty": float(lateral_penalty),
            "proximity_penalty": float(proximity_penalty),
            "attitude_penalty": float(attitude_penalty),
            "speed_penalty": float(speed_penalty),
            "collision_penalty": float(collision_penalty),
            "route_lateral_offset": float(route_lateral_offset),
            "goal_progress": float(goal_progress),
            "route_progress": float(route_progress),
        }
        return float(reward), reward_terms, goal_dist
