from __future__ import annotations

import numpy as np

from .geometry import distance_point_to_line_2d, route_projection_2d


def default_reward_terms():
    return {
        "progress_reward": 0.0,
        "distance_reward": 0.0,
        "time_penalty": 0.0,
        "goal_bonus": 0.0,
        "height_penalty": 0.0,
        "vertical_speed_penalty": 0.0,
        "lateral_penalty": 0.0,
        "proximity_penalty": 0.0,
        "attitude_penalty": 0.0,
        "speed_penalty": 0.0,
        "planar_speed": 0.0,
        "safe_speed": 0.0,
        "unsafe_speed_penalty": 0.0,
        "near_goal_speed_penalty": 0.0,
        "near_goal_away_penalty": 0.0,
        "near_goal_velocity_tracking_penalty": 0.0,
        "near_goal_desired_speed": 0.0,
        "near_goal_closing_speed": 0.0,
        "route_overshoot_penalty": 0.0,
        "collision_penalty": 0.0,
        "terminal_outcome_reward": 0.0,
        "safety_boundary_penalty": 0.0,
        "route_lateral_offset": 0.0,
        "goal_progress": 0.0,
        "route_progress": 0.0,
        "continuous_penalty_total": 0.0,
        "running_shaping_reward": 0.0,
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
        terminal_reason: str = "running",
    ):
        raise NotImplementedError


class BaselineForestReward(ForestRewardModel):
    def __init__(
        self,
        speed_penalty_weight: float = 0.003,
        safety_boundary_penalty_weight: float = 25.0,
        attitude_warning_rad: float = 0.50,
        attitude_limit_rad: float = 0.65,
        distance_reward_weight: float = 0.5,
        time_penalty_weight: float = 0.0,
    ):
        self.speed_penalty_weight = float(speed_penalty_weight)
        self.safety_boundary_penalty_weight = float(safety_boundary_penalty_weight)
        self.attitude_warning_rad = float(attitude_warning_rad)
        self.attitude_limit_rad = float(attitude_limit_rad)
        self.distance_reward_weight = float(distance_reward_weight)
        self.time_penalty_weight = float(time_penalty_weight)

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
        terminal_reason: str = "running",
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
        distance_reward = self.distance_reward_weight / (1.0 + goal_dist)
        time_penalty = self.time_penalty_weight
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
        speed_penalty = self.speed_penalty_weight * float(np.linalg.norm(vel))
        collision_penalty = 25.0 if collision else 0.0
        max_attitude = max(abs(float(roll)), abs(float(pitch)))
        attitude_boundary_ratio = np.clip(
            (max_attitude - self.attitude_warning_rad)
            / max(self.attitude_limit_rad - self.attitude_warning_rad, 1e-6),
            0.0,
            1.5,
        )
        height_boundary_ratio = max(
            np.clip((0.15 - float(pos[2])) / 0.07, 0.0, 1.5),
            np.clip((float(pos[2]) - 2.60) / 0.15, 0.0, 1.5),
        )
        safety_boundary_penalty = self.safety_boundary_penalty_weight * max(
            attitude_boundary_ratio,
            height_boundary_ratio,
        ) ** 2

        reward = progress_reward + distance_reward + goal_bonus
        reward -= height_penalty + vertical_speed_penalty + lateral_penalty
        reward -= proximity_penalty + attitude_penalty + speed_penalty + collision_penalty
        reward -= safety_boundary_penalty
        reward -= time_penalty

        reward_terms = {
            "progress_reward": float(progress_reward),
            "distance_reward": float(distance_reward),
            "time_penalty": float(time_penalty),
            "goal_bonus": float(goal_bonus),
            "height_penalty": float(height_penalty),
            "vertical_speed_penalty": float(vertical_speed_penalty),
            "lateral_penalty": float(lateral_penalty),
            "proximity_penalty": float(proximity_penalty),
            "attitude_penalty": float(attitude_penalty),
            "speed_penalty": float(speed_penalty),
            "collision_penalty": float(collision_penalty),
            "terminal_outcome_reward": 0.0,
            "safety_boundary_penalty": float(safety_boundary_penalty),
            "route_lateral_offset": float(route_lateral_offset),
            "goal_progress": float(goal_progress),
            "route_progress": float(route_progress),
        }
        return float(reward), reward_terms, goal_dist


class SafeTerminalForestReward(BaselineForestReward):
    """Safety-oriented reward that explicitly teaches goal capture and braking.

    The legacy reward intentionally remains unchanged for reproducibility.  In
    particular, its unbounded route projection can reward flying beyond the
    goal.  This variant caps route progress at the target, charges a small
    episode-time cost, and penalizes excessive/away-from-goal velocity near the
    terminal region.
    """

    def __init__(
        self,
        *,
        speed_penalty_weight: float = 0.003,
        safety_boundary_penalty_weight: float = 25.0,
        attitude_warning_rad: float = 0.50,
        attitude_limit_rad: float = 0.65,
        distance_reward_weight: float = 0.5,
        time_penalty_weight: float = 0.01,
        proximity_penalty_weight: float = 5.0,
        near_goal_radius: float = 0.8,
        near_goal_speed_penalty_weight: float = 1.5,
        near_goal_away_penalty_weight: float = 3.0,
        route_overshoot_penalty_weight: float = 1.0,
    ):
        super().__init__(
            speed_penalty_weight=speed_penalty_weight,
            safety_boundary_penalty_weight=safety_boundary_penalty_weight,
            attitude_warning_rad=attitude_warning_rad,
            attitude_limit_rad=attitude_limit_rad,
            distance_reward_weight=distance_reward_weight,
            time_penalty_weight=time_penalty_weight,
        )
        if min(
            proximity_penalty_weight,
            near_goal_radius,
            near_goal_speed_penalty_weight,
            near_goal_away_penalty_weight,
            route_overshoot_penalty_weight,
        ) < 0.0:
            raise ValueError("Safe-terminal reward weights and radius must be non-negative.")
        self.proximity_penalty_weight = float(proximity_penalty_weight)
        self.near_goal_radius = float(near_goal_radius)
        self.near_goal_speed_penalty_weight = float(near_goal_speed_penalty_weight)
        self.near_goal_away_penalty_weight = float(near_goal_away_penalty_weight)
        self.route_overshoot_penalty_weight = float(route_overshoot_penalty_weight)

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
        terminal_reason: str = "running",
    ):
        state = np.asarray(state, dtype=np.float32)
        pos = state[0:3]
        roll, pitch = state[7], state[8]
        vel = state[10:13]
        start_pos = np.asarray(start_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        prev_pos = np.asarray(prev_pos, dtype=np.float32)

        goal_vector = target_pos - pos
        goal_dist = float(np.linalg.norm(goal_vector))
        reference_goal_dist = goal_dist if prev_goal_dist is None else float(prev_goal_dist)
        goal_progress = reference_goal_dist - goal_dist

        route_length = float(np.linalg.norm(target_pos[:2] - start_pos[:2]))
        current_route_projection = route_projection_2d(pos[:2], start_pos[:2], target_pos[:2])
        previous_route_projection = route_projection_2d(
            prev_pos[:2], start_pos[:2], target_pos[:2]
        )
        # This is the key terminal fix: no positive route-progress reward is
        # available after passing the goal plane.
        capped_current_projection = float(np.clip(current_route_projection, 0.0, route_length))
        capped_previous_projection = float(np.clip(previous_route_projection, 0.0, route_length))
        route_progress = capped_current_projection - capped_previous_projection
        route_overshoot = max(0.0, current_route_projection - route_length)

        progress_reward = 35.0 * route_progress + 15.0 * goal_progress
        distance_reward = self.distance_reward_weight / (1.0 + goal_dist)
        time_penalty = self.time_penalty_weight
        goal_bonus = 120.0 if goal_dist < goal_tolerance else 0.0

        target_z = float(target_pos[2])
        route_lateral_offset = distance_point_to_line_2d(pos[:2], start_pos[:2], target_pos[:2])
        height_penalty = 0.8 * abs(float(pos[2]) - target_z)
        vertical_speed_penalty = 0.12 * abs(float(vel[2]))
        lateral_penalty = 0.15 * route_lateral_offset

        proximity_penalty = 0.0
        if clearance < safe_distance:
            proximity_penalty = self.proximity_penalty_weight * (
                (safe_distance - clearance) / max(safe_distance, 1e-6)
            ) ** 2

        near_goal_speed_penalty = 0.0
        near_goal_away_penalty = 0.0
        if self.near_goal_radius > 0.0 and goal_dist < self.near_goal_radius:
            terminal_weight = 1.0 - goal_dist / self.near_goal_radius
            speed = float(np.linalg.norm(vel))
            near_goal_speed_penalty = (
                self.near_goal_speed_penalty_weight * terminal_weight * speed**2
            )
            if goal_dist > 1e-6:
                closing_speed = float(np.dot(vel, goal_vector / goal_dist))
                near_goal_away_penalty = (
                    self.near_goal_away_penalty_weight
                    * terminal_weight
                    * max(0.0, -closing_speed)
                )
        route_overshoot_penalty = self.route_overshoot_penalty_weight * route_overshoot**2

        attitude_penalty = 0.10 * (abs(roll) + abs(pitch))
        speed_penalty = self.speed_penalty_weight * float(np.linalg.norm(vel))
        collision_penalty = 25.0 if collision else 0.0
        max_attitude = max(abs(float(roll)), abs(float(pitch)))
        attitude_boundary_ratio = np.clip(
            (max_attitude - self.attitude_warning_rad)
            / max(self.attitude_limit_rad - self.attitude_warning_rad, 1e-6),
            0.0,
            1.5,
        )
        height_boundary_ratio = max(
            np.clip((0.15 - float(pos[2])) / 0.07, 0.0, 1.5),
            np.clip((float(pos[2]) - 2.60) / 0.15, 0.0, 1.5),
        )
        safety_boundary_penalty = self.safety_boundary_penalty_weight * max(
            attitude_boundary_ratio,
            height_boundary_ratio,
        ) ** 2

        reward = progress_reward + distance_reward + goal_bonus
        reward -= height_penalty + vertical_speed_penalty + lateral_penalty
        reward -= proximity_penalty + attitude_penalty + speed_penalty + collision_penalty
        reward -= near_goal_speed_penalty + near_goal_away_penalty + route_overshoot_penalty
        reward -= safety_boundary_penalty + time_penalty

        reward_terms = {
            "progress_reward": float(progress_reward),
            "distance_reward": float(distance_reward),
            "time_penalty": float(time_penalty),
            "goal_bonus": float(goal_bonus),
            "height_penalty": float(height_penalty),
            "vertical_speed_penalty": float(vertical_speed_penalty),
            "lateral_penalty": float(lateral_penalty),
            "proximity_penalty": float(proximity_penalty),
            "attitude_penalty": float(attitude_penalty),
            "speed_penalty": float(speed_penalty),
            "near_goal_speed_penalty": float(near_goal_speed_penalty),
            "near_goal_away_penalty": float(near_goal_away_penalty),
            "route_overshoot_penalty": float(route_overshoot_penalty),
            "collision_penalty": float(collision_penalty),
            "terminal_outcome_reward": 0.0,
            "safety_boundary_penalty": float(safety_boundary_penalty),
            "route_lateral_offset": float(route_lateral_offset),
            "goal_progress": float(goal_progress),
            "route_progress": float(route_progress),
        }
        return float(reward), reward_terms, goal_dist


class OutcomeSeparatedForestReward(SafeTerminalForestReward):
    """Progress shaping plus mutually exclusive terminal outcomes.

    Absolute distance is deliberately absent: approaching the goal is learned
    from potential differences, while success and each failure mode emit one
    non-accumulating terminal reward.
    """

    def __init__(
        self,
        *,
        success_terminal_reward: float = 150.0,
        collision_terminal_penalty: float = 120.0,
        bound_terminal_penalty: float = 100.0,
        timeout_terminal_penalty: float = 30.0,
        **kwargs,
    ):
        kwargs["distance_reward_weight"] = 0.0
        super().__init__(**kwargs)
        values = (
            success_terminal_reward,
            collision_terminal_penalty,
            bound_terminal_penalty,
            timeout_terminal_penalty,
        )
        if min(values) < 0.0:
            raise ValueError("Terminal reward magnitudes must be non-negative.")
        self.success_terminal_reward = float(success_terminal_reward)
        self.collision_terminal_penalty = float(collision_terminal_penalty)
        self.bound_terminal_penalty = float(bound_terminal_penalty)
        self.timeout_terminal_penalty = float(timeout_terminal_penalty)

    def compute(self, **kwargs):
        _, shaped_terms, goal_dist = super().compute(**kwargs)
        terminal_reason = str(kwargs.get("terminal_reason", "running"))
        if terminal_reason == "running":
            shaped_reward = (
                shaped_terms["progress_reward"]
                - shaped_terms["time_penalty"]
                - shaped_terms["height_penalty"]
                - shaped_terms["vertical_speed_penalty"]
                - shaped_terms["lateral_penalty"]
                - shaped_terms["proximity_penalty"]
                - shaped_terms["attitude_penalty"]
                - shaped_terms["speed_penalty"]
                - shaped_terms["near_goal_speed_penalty"]
                - shaped_terms["near_goal_away_penalty"]
                - shaped_terms["route_overshoot_penalty"]
                - shaped_terms["safety_boundary_penalty"]
            )
            shaped_terms["distance_reward"] = 0.0
            shaped_terms["goal_bonus"] = 0.0
            shaped_terms["collision_penalty"] = 0.0
            shaped_terms["terminal_outcome_reward"] = 0.0
            return float(shaped_reward), shaped_terms, goal_dist

        reward_terms = default_reward_terms()
        if terminal_reason == "success":
            terminal_reward = self.success_terminal_reward
            reward_terms["goal_bonus"] = terminal_reward
        elif terminal_reason == "collision":
            terminal_reward = -self.collision_terminal_penalty
            reward_terms["collision_penalty"] = self.collision_terminal_penalty
        elif terminal_reason == "timeout":
            terminal_reward = -self.timeout_terminal_penalty
        else:
            terminal_reward = -self.bound_terminal_penalty
        reward_terms["terminal_outcome_reward"] = float(terminal_reward)
        reward_terms["goal_progress"] = float(shaped_terms["goal_progress"])
        reward_terms["route_progress"] = float(shaped_terms["route_progress"])
        reward_terms["route_lateral_offset"] = float(
            shaped_terms["route_lateral_offset"]
        )
        return float(terminal_reward), reward_terms, goal_dist


class RateIntegratedOutcomeForestReward(OutcomeSeparatedForestReward):
    """Frequency-invariant upper-policy reward with separated outcomes.

    Progress remains a potential difference, so its episode sum is independent
    of how many 120 Hz lower-control ticks execute.  State-dependent costs are
    interpreted as rates and integrated with ``integration_dt``.  This avoids
    multiplying obstacle, attitude, and near-goal costs by the lower-controller
    frequency before an upper transition is stored.

    The straight-line lateral cost can be suppressed independently.  That is
    important in obstacle fields, where a safe detour must leave the direct
    start-to-goal line.
    """

    _CONTINUOUS_PENALTY_KEYS = (
        "time_penalty",
        "height_penalty",
        "vertical_speed_penalty",
        "lateral_penalty",
        "proximity_penalty",
        "attitude_penalty",
        "speed_penalty",
        "near_goal_speed_penalty",
        "near_goal_away_penalty",
        "route_overshoot_penalty",
        "safety_boundary_penalty",
    )

    def __init__(
        self,
        *,
        integration_dt: float,
        route_progress_reward_weight: float = 12.0,
        goal_progress_reward_weight: float = 8.0,
        lateral_penalty_scale: float = 0.0,
        **kwargs,
    ):
        if integration_dt <= 0.0:
            raise ValueError("integration_dt must be positive.")
        if min(
            route_progress_reward_weight,
            goal_progress_reward_weight,
            lateral_penalty_scale,
        ) < 0.0:
            raise ValueError("Rate-integrated reward scales must be non-negative.")
        super().__init__(**kwargs)
        self.integration_dt = float(integration_dt)
        self.route_progress_reward_weight = float(route_progress_reward_weight)
        self.goal_progress_reward_weight = float(goal_progress_reward_weight)
        self.lateral_penalty_scale = float(lateral_penalty_scale)

    def compute(self, **kwargs):
        reward, reward_terms, goal_dist = super().compute(**kwargs)
        reward_terms.setdefault("near_goal_velocity_tracking_penalty", 0.0)
        reward_terms.setdefault("near_goal_desired_speed", 0.0)
        reward_terms.setdefault("near_goal_closing_speed", 0.0)
        terminal_reason = str(kwargs.get("terminal_reason", "running"))
        if terminal_reason != "running":
            reward_terms["continuous_penalty_total"] = 0.0
            reward_terms["running_shaping_reward"] = 0.0
            return reward, reward_terms, goal_dist

        reward_terms["progress_reward"] = (
            self.route_progress_reward_weight * reward_terms["route_progress"]
            + self.goal_progress_reward_weight * reward_terms["goal_progress"]
        )

        continuous_penalty_total = 0.0
        for key in self._CONTINUOUS_PENALTY_KEYS:
            scale = self.lateral_penalty_scale if key == "lateral_penalty" else 1.0
            integrated_value = float(reward_terms[key]) * scale * self.integration_dt
            reward_terms[key] = integrated_value
            continuous_penalty_total += integrated_value

        shaped_reward = reward_terms["progress_reward"] - continuous_penalty_total
        reward_terms["distance_reward"] = 0.0
        reward_terms["goal_bonus"] = 0.0
        reward_terms["collision_penalty"] = 0.0
        reward_terms["terminal_outcome_reward"] = 0.0
        reward_terms["continuous_penalty_total"] = float(continuous_penalty_total)
        reward_terms["running_shaping_reward"] = float(shaped_reward)
        return float(shaped_reward), reward_terms, goal_dist


class TerminalGuidanceRateIntegratedReward(RateIntegratedOutcomeForestReward):
    """Rate-integrated outcome reward with directed terminal capture guidance.

    The V4 near-goal speed cost only says "move less".  It cannot distinguish
    a slow action toward the goal from tangential loitering, and its optimum is
    zero velocity before the success sphere.  This variant instead defines a
    feasible desired velocity that points at the 3-D goal.  Its magnitude
    decreases from cruise speed at ``guidance_radius`` to a small non-zero
    capture speed at the success radius.  The non-zero terminal speed keeps
    the policy moving into the unchanged success sphere rather than stopping
    immediately outside it.

    The tracking cost is a physical rate and is integrated using the same
    decision interval as the other V4 continuous costs.  Terminal rewards and
    all task termination conditions remain unchanged.
    """

    def __init__(
        self,
        *,
        near_goal_guidance_radius: float = 1.2,
        near_goal_guidance_ramp_width: float = 0.2,
        near_goal_capture_speed: float = 0.1,
        near_goal_cruise_speed: float = 0.8,
        near_goal_velocity_tracking_weight: float = 160.0,
        **kwargs,
    ):
        values = (
            near_goal_guidance_radius,
            near_goal_guidance_ramp_width,
            near_goal_capture_speed,
            near_goal_cruise_speed,
            near_goal_velocity_tracking_weight,
        )
        if min(values) < 0.0:
            raise ValueError("Terminal-guidance parameters must be non-negative.")
        if near_goal_guidance_radius <= 0.0:
            raise ValueError("near_goal_guidance_radius must be positive.")
        if near_goal_guidance_ramp_width <= 0.0:
            raise ValueError("near_goal_guidance_ramp_width must be positive.")
        if near_goal_capture_speed > near_goal_cruise_speed:
            raise ValueError(
                "near_goal_capture_speed cannot exceed near_goal_cruise_speed."
            )
        super().__init__(**kwargs)
        self.near_goal_guidance_radius = float(near_goal_guidance_radius)
        self.near_goal_guidance_ramp_width = float(
            near_goal_guidance_ramp_width
        )
        self.near_goal_capture_speed = float(near_goal_capture_speed)
        self.near_goal_cruise_speed = float(near_goal_cruise_speed)
        self.near_goal_velocity_tracking_weight = float(
            near_goal_velocity_tracking_weight
        )

    def compute(self, **kwargs):
        reward, reward_terms, goal_dist = super().compute(**kwargs)
        terminal_reason = str(kwargs.get("terminal_reason", "running"))
        if terminal_reason != "running":
            return reward, reward_terms, goal_dist

        goal_tolerance = float(kwargs["goal_tolerance"])
        if self.near_goal_guidance_radius <= goal_tolerance:
            raise ValueError(
                "near_goal_guidance_radius must exceed goal_tolerance."
            )
        if goal_dist >= self.near_goal_guidance_radius or goal_dist <= 1e-6:
            return reward, reward_terms, goal_dist

        state = np.asarray(kwargs["state"], dtype=np.float32)
        target_pos = np.asarray(kwargs["target_pos"], dtype=np.float32)
        position = state[0:3]
        velocity = state[10:13]
        goal_vector = target_pos - position
        goal_direction = goal_vector / max(goal_dist, 1e-6)

        distance_ratio = float(
            np.clip(
                (goal_dist - goal_tolerance)
                / (self.near_goal_guidance_radius - goal_tolerance),
                0.0,
                1.0,
            )
        )
        desired_speed = self.near_goal_capture_speed + distance_ratio * (
            self.near_goal_cruise_speed - self.near_goal_capture_speed
        )
        desired_velocity = desired_speed * goal_direction
        velocity_error = velocity - desired_velocity
        activation = float(
            np.clip(
                (self.near_goal_guidance_radius - goal_dist)
                / self.near_goal_guidance_ramp_width,
                0.0,
                1.0,
            )
        )
        raw_tracking_penalty = (
            self.near_goal_velocity_tracking_weight
            * activation
            * float(np.dot(velocity_error, velocity_error))
        )
        tracking_penalty = raw_tracking_penalty * self.integration_dt
        closing_speed = float(np.dot(velocity, goal_direction))

        reward_terms["near_goal_velocity_tracking_penalty"] = float(
            tracking_penalty
        )
        reward_terms["near_goal_desired_speed"] = float(desired_speed)
        reward_terms["near_goal_closing_speed"] = float(closing_speed)
        reward_terms["continuous_penalty_total"] += float(tracking_penalty)
        reward_terms["running_shaping_reward"] -= float(tracking_penalty)
        return float(reward - tracking_penalty), reward_terms, goal_dist


class MAVRLSafeSpeedForestReward(SafeTerminalForestReward):
    """Safe-terminal reward with MAVRL-style clearance-conditioned speed limits.

    Unlike the legacy global speed cost, this variant leaves motion below a
    clearance-dependent safe planar speed unpenalized.  The policy can
    therefore gain time in open space while still learning to brake before
    entering low-clearance regions.
    """

    def __init__(
        self,
        *,
        safe_speed_min: float = 0.0,
        safe_speed_max: float = 0.8,
        safe_speed_brake_clearance: float = 0.35,
        safe_speed_open_clearance: float = 1.20,
        unsafe_speed_penalty_weight: float = 3.0,
        **kwargs,
    ):
        if min(
            safe_speed_min,
            safe_speed_max,
            safe_speed_brake_clearance,
            safe_speed_open_clearance,
            unsafe_speed_penalty_weight,
        ) < 0.0:
            raise ValueError("MAVRL safe-speed parameters must be non-negative.")
        if safe_speed_max < safe_speed_min:
            raise ValueError("safe_speed_max must be at least safe_speed_min.")
        if safe_speed_open_clearance <= safe_speed_brake_clearance:
            raise ValueError(
                "safe_speed_open_clearance must exceed safe_speed_brake_clearance."
            )

        # MAVRL-style speed adaptation is incompatible with a global speed
        # penalty: the latter would reward slow motion even in open space.
        kwargs["speed_penalty_weight"] = 0.0
        super().__init__(**kwargs)
        self.safe_speed_min = float(safe_speed_min)
        self.safe_speed_max = float(safe_speed_max)
        self.safe_speed_brake_clearance = float(safe_speed_brake_clearance)
        self.safe_speed_open_clearance = float(safe_speed_open_clearance)
        self.unsafe_speed_penalty_weight = float(unsafe_speed_penalty_weight)

    def _safe_speed(self, clearance: float) -> float:
        interpolation = np.clip(
            (float(clearance) - self.safe_speed_brake_clearance)
            / (self.safe_speed_open_clearance - self.safe_speed_brake_clearance),
            0.0,
            1.0,
        )
        return float(
            self.safe_speed_min
            + interpolation * (self.safe_speed_max - self.safe_speed_min)
        )

    def compute(self, **kwargs):
        reward, reward_terms, goal_dist = super().compute(**kwargs)
        state = np.asarray(kwargs["state"], dtype=np.float32)
        planar_speed = float(np.linalg.norm(state[10:12]))
        safe_speed = self._safe_speed(float(kwargs["clearance"]))
        unsafe_speed_penalty = self.unsafe_speed_penalty_weight * max(
            0.0, planar_speed - safe_speed
        ) ** 2

        reward -= unsafe_speed_penalty
        reward_terms["planar_speed"] = planar_speed
        reward_terms["safe_speed"] = safe_speed
        reward_terms["unsafe_speed_penalty"] = float(unsafe_speed_penalty)
        return float(reward), reward_terms, goal_dist
