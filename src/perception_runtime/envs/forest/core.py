from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

from envs.observation_layout import (
    GOAL_DIM,
    RANGE_DIM,
    ForestObservationLayout,
    append_forest_task_observation,
)
from envs.map_atlas import MapRuntime, coerce_map_spec

from .curriculum import DEFAULT_CURRICULUM_MILESTONES, ForestCurriculumScheduler
from .layout import ForestLayoutConfig, ForestLayoutGenerator
from .rewards import BaselineForestReward, ForestRewardModel, default_reward_terms


class CustomForestAviary(BaseRLAviary):
    """Single-agent RL environment for goal reaching with forest obstacle avoidance."""

    DEFAULT_START_POS = np.array([-3.5, 0.0, 1.0], dtype=np.float32)
    DEFAULT_TARGET_POS = np.array([3.5, 0.0, 1.0], dtype=np.float32)
    DEFAULT_CURRICULUM_MILESTONES = DEFAULT_CURRICULUM_MILESTONES

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        target_pos=None,
        map_spec=None,
        forest_half_extent: float = 3.0,
        num_trees: int = 24,
        tree_radius_range=(0.10, 0.22),
        tree_height_range=(1.2, 2.4),
        min_tree_separation: float = 0.45,
        corridor_half_width: float = 0.55,
        obstacle_sensor_range: float = 1.75,
        range_ray_origin_offset: float = 0.08,
        range_azimuth_count: int = RANGE_DIM,
        range_elevation_degrees=(0.0,),
        episode_len_sec: float = 12.0,
        safe_distance: float = 0.35,
        goal_tolerance: float = 0.20,
        curriculum: bool = True,
        curriculum_milestones=DEFAULT_CURRICULUM_MILESTONES,
        curriculum_success_gated: bool = False,
        curriculum_success_window: int = 100,
        curriculum_success_thresholds=(0.10, 0.20, 0.30),
        curriculum_max_collision_rates=(1.0, 1.0, 1.0),
        curriculum_minimum_stage_episodes: int = 50,
        wide_corridor_half_width: float = 1.35,
        narrow_corridor_half_width: float = 0.35,
        centerline_tree_fraction: float = 0.35,
        centerline_band_width: float = 0.40,
        route_blocking_tree: bool = True,
        route_tree_fraction: float = 0.5,
        route_blocking_tree_count: int = 1,
        route_tree_lateral_range: float = 0.55,
        route_tree_layout: str = "random",
        route_count_curriculum: bool = False,
        route_count_curriculum_counts=(1, 2, 3, 5),
        curriculum_simple_mix_probability: float = 0.20,
        seed: int | None = None,
        reward_model: ForestRewardModel | None = None,
        curriculum_scheduler: ForestCurriculumScheduler | None = None,
    ):
        self.MAP_SPEC = coerce_map_spec(map_spec)
        map_overrides = (
            dict(self.MAP_SPEC.environment_overrides) if self.MAP_SPEC is not None else {}
        )
        if self.MAP_SPEC is not None:
            # A selected atlas map owns its task frame. This prevents a caller
            # from accidentally combining the geometry of one map with the
            # start/goal pair of another.
            initial_xyzs = np.asarray([self.MAP_SPEC.start_pos], dtype=np.float32)
            target_pos = self.MAP_SPEC.target_pos
            forest_half_extent = self.MAP_SPEC.xy_half_extent
            obstacle_sensor_range = self.MAP_SPEC.obstacle_sensor_range
            if self.MAP_SPEC.episode_len_sec is not None:
                episode_len_sec = self.MAP_SPEC.episode_len_sec
            num_trees = map_overrides.get("num_trees", num_trees)
            route_blocking_tree = map_overrides.get(
                "route_blocking_tree", route_blocking_tree
            )
            route_blocking_tree_count = map_overrides.get(
                "route_blocking_tree_count", route_blocking_tree_count
            )
            route_tree_layout = map_overrides.get("route_tree_layout", route_tree_layout)
            curriculum = map_overrides.get("curriculum", curriculum)
            route_count_curriculum = map_overrides.get(
                "route_count_curriculum", route_count_curriculum
            )
            route_count_curriculum_counts = map_overrides.get(
                "route_count_curriculum_counts", route_count_curriculum_counts
            )
            curriculum_simple_mix_probability = map_overrides.get(
                "curriculum_simple_mix_probability",
                curriculum_simple_mix_probability,
            )

        default_initial_xyzs = np.array([self.DEFAULT_START_POS], dtype=np.float32)
        if initial_xyzs is None:
            initial_xyzs = default_initial_xyzs.copy()

        self.START_POS = np.array(initial_xyzs[0], dtype=np.float32)
        self.TARGET_POS = np.array(
            target_pos if target_pos is not None else self.DEFAULT_TARGET_POS,
            dtype=np.float32,
        )
        self.EPISODE_LEN_SEC = float(episode_len_sec)
        self.FOREST_HALF_EXTENT = float(forest_half_extent)
        self.NUM_TREES = int(num_trees)
        self.TREE_RADIUS_RANGE = (float(tree_radius_range[0]), float(tree_radius_range[1]))
        self.TREE_HEIGHT_RANGE = (float(tree_height_range[0]), float(tree_height_range[1]))
        self.MIN_TREE_SEPARATION = float(min_tree_separation)
        self.CORRIDOR_HALF_WIDTH = float(corridor_half_width)
        self.OBSTACLE_SENSOR_RANGE = float(obstacle_sensor_range)
        self.RANGE_RAY_ORIGIN_OFFSET = float(range_ray_origin_offset)
        if not 0.0 <= self.RANGE_RAY_ORIGIN_OFFSET < self.OBSTACLE_SENSOR_RANGE:
            raise ValueError(
                "range_ray_origin_offset must lie in [0, obstacle_sensor_range)."
            )
        self.SAFE_DISTANCE = float(safe_distance)
        self.GOAL_TOLERANCE = float(goal_tolerance)
        self.CURRICULUM_ENABLED = bool(curriculum)
        self.CURRICULUM_MILESTONES = tuple(int(x) for x in curriculum_milestones)
        self.WIDE_CORRIDOR_HALF_WIDTH = float(max(wide_corridor_half_width, self.CORRIDOR_HALF_WIDTH))
        self.NARROW_CORRIDOR_HALF_WIDTH = float(min(narrow_corridor_half_width, self.CORRIDOR_HALF_WIDTH))
        self.CENTERLINE_TREE_FRACTION = float(np.clip(centerline_tree_fraction, 0.0, 1.0))
        self.CENTERLINE_BAND_WIDTH = float(centerline_band_width)
        self.ROUTE_BLOCKING_TREE = bool(route_blocking_tree)
        self.ROUTE_TREE_FRACTION = float(np.clip(route_tree_fraction, 0.05, 0.95))
        self.ROUTE_BLOCKING_TREE_COUNT = max(0, int(route_blocking_tree_count))
        self.ROUTE_TREE_LATERAL_RANGE = max(0.0, float(route_tree_lateral_range))
        self.ROUTE_TREE_LAYOUT = str(route_tree_layout)
        self.ROUTE_COUNT_CURRICULUM = bool(route_count_curriculum)
        self.ROUTE_COUNT_CURRICULUM_COUNTS = tuple(
            max(0, int(value)) for value in route_count_curriculum_counts
        )
        if len(self.ROUTE_COUNT_CURRICULUM_COUNTS) != 4:
            raise ValueError("route_count_curriculum_counts must contain four stage counts.")
        self.CURRICULUM_SIMPLE_MIX_PROBABILITY = float(
            curriculum_simple_mix_probability
        )
        if not 0.0 <= self.CURRICULUM_SIMPLE_MIX_PROBABILITY <= 1.0:
            raise ValueError("curriculum_simple_mix_probability must lie in [0, 1].")

        self.RANGE_AZIMUTH_COUNT = int(range_azimuth_count)
        self.RANGE_ELEVATION_DEGREES = tuple(
            float(value) for value in range_elevation_degrees
        )
        if self.RANGE_AZIMUTH_COUNT <= 0 or not self.RANGE_ELEVATION_DEGREES:
            raise ValueError("Range-ray azimuth and elevation counts must be positive.")
        self.NUM_RANGE_RAYS = self.RANGE_AZIMUTH_COUNT * len(
            self.RANGE_ELEVATION_DEGREES
        )
        self._range_directions = self._buildRangeDirections()
        self.GOAL_OBS_DIM = GOAL_DIM
        self.OBSTACLE_OBS_DIM = self.GOAL_OBS_DIM + self.NUM_RANGE_RAYS

        self._rng = np.random.default_rng(seed)
        self._tree_ids = []
        self._tree_specs = []
        self._map_runtime = None
        self._prev_goal_dist = None
        self._prev_pos = self.START_POS.copy()
        self._last_collision = False
        self._last_clearance = np.inf
        self._last_reward_terms = {}
        self._last_done_reason = "running"
        self.motor_action_codec = None
        self._last_motor_rpm = None
        self._replay_size = 0
        self._reset_count = 0
        self._curriculum_stage = 0
        self._curriculum_stage_override = None
        self._current_corridor_half_width = self.WIDE_CORRIDOR_HALF_WIDTH
        self._protect_corridor = True
        self._corridor_edge_tree_fraction = 0.0
        self._centerline_bias_fraction = 0.0
        self._active_route_blocking_tree_count = self.ROUTE_BLOCKING_TREE_COUNT
        self._curriculum_mixed_difficulty = False

        self._layout_generator = ForestLayoutGenerator(
            ForestLayoutConfig(
                forest_half_extent=self.FOREST_HALF_EXTENT,
                num_trees=self.NUM_TREES,
                tree_radius_range=self.TREE_RADIUS_RANGE,
                tree_height_range=self.TREE_HEIGHT_RANGE,
                min_tree_separation=self.MIN_TREE_SEPARATION,
                centerline_band_width=self.CENTERLINE_BAND_WIDTH,
                route_blocking_tree=self.ROUTE_BLOCKING_TREE,
                route_tree_fraction=self.ROUTE_TREE_FRACTION,
                route_blocking_tree_count=self.ROUTE_BLOCKING_TREE_COUNT,
                route_tree_lateral_range=self.ROUTE_TREE_LATERAL_RANGE,
                route_tree_layout=self.ROUTE_TREE_LAYOUT,
                xy_half_extents=(
                    self.MAP_SPEC.xy_half_extents if self.MAP_SPEC is not None else None
                ),
            )
        )
        self.map_validation = (
            self.MAP_SPEC.validate_route() if self.MAP_SPEC is not None else None
        )
        if self.map_validation is not None and self.map_validation.reachable is False:
            raise ValueError(
                f"Map {self.MAP_SPEC.map_id!r} cannot be used: {self.map_validation.message}."
            )
        self._reward_model = reward_model or BaselineForestReward()
        self._curriculum_scheduler = curriculum_scheduler or ForestCurriculumScheduler(
            enabled=self.CURRICULUM_ENABLED,
            milestones=self.CURRICULUM_MILESTONES,
            corridor_half_width=self.CORRIDOR_HALF_WIDTH,
            wide_corridor_half_width=self.WIDE_CORRIDOR_HALF_WIDTH,
            narrow_corridor_half_width=self.NARROW_CORRIDOR_HALF_WIDTH,
            centerline_tree_fraction=self.CENTERLINE_TREE_FRACTION,
            success_gated=curriculum_success_gated,
            success_window=curriculum_success_window,
            success_thresholds=curriculum_success_thresholds,
            max_collision_rates=curriculum_max_collision_rates,
            minimum_stage_episodes=curriculum_minimum_stage_episodes,
        )
        self._applyCurriculum()

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_count += 1
        self._applyCurriculum()
        obs, info = super().reset(seed=seed, options=options)
        self._resetActionBuffer()

        obs = self._computeObs()
        state = self._getDroneStateVector(0)
        self._prev_goal_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._prev_pos = np.array(state[0:3], dtype=np.float32)
        self._last_collision = self._checkCollision()
        self._last_clearance = self._computeNearestTreeClearance(state[0:3])
        self._last_reward_terms = default_reward_terms()
        self._last_done_reason = "running"
        self._last_motor_rpm = None

        info = self._computeInfo()
        return obs, info

    def set_motor_action_codec(self, codec):
        """Use an explicit normalized-action/RPM codec for direct motor control."""

        self.motor_action_codec = codec
        self._last_motor_rpm = None

    def _preprocessAction(self, action):
        if self.ACT_TYPE != ActionType.RPM or self.motor_action_codec is None:
            return super()._preprocessAction(action)

        action = np.asarray(action, dtype=np.float64).reshape(self.NUM_DRONES, 4)
        self.action_buffer.append(action.astype(np.float32))
        rpm = np.vstack(
            [
                self.motor_action_codec.normalized_action_to_rpm(
                    motor_action,
                    previous_rpm=(
                        None if self._last_motor_rpm is None else self._last_motor_rpm[index]
                    ),
                )
                for index, motor_action in enumerate(action)
            ]
        )
        self._last_motor_rpm = rpm.copy()
        return rpm

    def _resetActionBuffer(self):
        action_dim = int(self.action_space.shape[-1])
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, action_dim), dtype=np.float32))

    def _addObstacles(self):
        if self.OBS_TYPE == ObservationType.RGB:
            super()._addObstacles()
            return

        self._tree_ids = []
        self._tree_specs = []
        self._map_runtime = None
        if self.MAP_SPEC is not None and self.MAP_SPEC.obstacles:
            self._map_runtime = MapRuntime(self.MAP_SPEC)
            self._tree_ids = self._map_runtime.create_bodies(client_id=self.CLIENT)
            self._tree_specs = [
                {"kind": obstacle.kind, "name": obstacle.name, "id": body_id}
                for obstacle, body_id in zip(self.MAP_SPEC.obstacles, self._tree_ids)
            ]
            return
        tree_specs = self._layout_generator.generate(
            rng=self._rng,
            start_pos=self.INIT_XYZS[0],
            goal_pos=self.TARGET_POS,
            corridor_half_width=self._current_corridor_half_width,
            protect_corridor=self._protect_corridor,
            corridor_edge_tree_fraction=self._corridor_edge_tree_fraction,
            centerline_tree_fraction=self._centerline_bias_fraction,
            route_blocking_tree_count=self._active_route_blocking_tree_count,
        )

        for spec in tree_specs:
            tree_id = self._layout_generator.create_tree_body(
                client_id=self.CLIENT,
                xy=spec["xy"],
                radius=spec["radius"],
                height=spec["height"],
            )
            self._tree_ids.append(tree_id)
            self._tree_specs.append(
                {
                    **spec,
                    "id": tree_id,
                }
            )

    def _observationSpace(self):
        base_space = super()._observationSpace()
        if self.OBS_TYPE != ObservationType.KIN:
            return base_space

        extra_low = np.array([[-1.0, -1.0, -1.0] + [0.0] * self.NUM_RANGE_RAYS], dtype=np.float32)
        extra_high = np.array([[1.0, 1.0, 1.0] + [1.0] * self.NUM_RANGE_RAYS], dtype=np.float32)
        goal_low = extra_low[..., : self.GOAL_OBS_DIM]
        range_low = extra_low[..., self.GOAL_OBS_DIM :]
        goal_high = extra_high[..., : self.GOAL_OBS_DIM]
        range_high = extra_high[..., self.GOAL_OBS_DIM :]
        low = append_forest_task_observation(
            base_space.low,
            goal_low,
            range_low,
            expected_range_dim=self.NUM_RANGE_RAYS,
        )
        high = append_forest_task_observation(
            base_space.high,
            goal_high,
            range_high,
            expected_range_dim=self.NUM_RANGE_RAYS,
        )
        self.observation_layout = ForestObservationLayout.from_total_dim(
            low.shape[-1],
            action_dim=self.action_space.shape[-1],
            range_dim=self.NUM_RANGE_RAYS,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        base_obs = super()._computeObs()
        if self.OBS_TYPE != ObservationType.KIN:
            return base_obs

        state = self._getDroneStateVector(0)
        goal_obs = self._computeGoalObservation(state[0:3]).reshape(1, -1)
        range_obs = self._computeRangeObservation(state[0:3]).reshape(1, -1)
        obs = append_forest_task_observation(
            base_obs,
            goal_obs,
            range_obs,
            expected_range_dim=self.NUM_RANGE_RAYS,
        ).astype(np.float32)
        self.observation_layout.split(obs)
        return obs

    def _computeGoalObservation(self, pos):
        rel = (self.TARGET_POS - np.array(pos, dtype=np.float32)).astype(np.float32)
        xy_scale = max(
            self.FOREST_HALF_EXTENT,
            float(np.max(np.abs(self.START_POS[:2]))),
            float(np.max(np.abs(self.TARGET_POS[:2]))),
        )
        scale = np.array(
            [xy_scale, xy_scale, max(1.0, float(max(self.START_POS[2], self.TARGET_POS[2])) + 1.0)],
            dtype=np.float32,
        )
        return np.clip(rel / scale, -1.0, 1.0)

    def _rangeRayGeometry(self, pos):
        """Return the self-hit-free rays and their centre-range conversion data.

        This is deliberately kept separate from observation conversion so that
        the sensor contract can be regression-tested against PyBullet's raw
        body ids: no ray may start inside and immediately hit the UAV itself.
        """
        # A ray starting at the vehicle centre intersects the drone's own
        # collision mesh, especially for downward directions.  Begin each ray
        # outside the collision envelope instead, then convert a hit distance
        # back to the centre-referenced [0, 1] sensor range below.
        origin = np.array(pos, dtype=np.float32) + np.array([0.0, 0.0, 0.02], dtype=np.float32)
        origin_offset = self.RANGE_RAY_ORIGIN_OFFSET
        ray_from = origin.reshape(1, 3) + self._range_directions * origin_offset
        ray_to = origin.reshape(1, 3) + self._range_directions * self.OBSTACLE_SENSOR_RANGE
        cast_length = self.OBSTACLE_SENSOR_RANGE - origin_offset
        return ray_from, ray_to, origin_offset, cast_length

    def _computeRangeObservation(self, pos):
        ray_from, ray_to, origin_offset, cast_length = self._rangeRayGeometry(pos)
        hits = p.rayTestBatch(ray_from.tolist(), ray_to.tolist(), physicsClientId=self.CLIENT)

        dists = []
        for hit in hits:
            hit_fraction = float(hit[2])
            if hit_fraction < 0.0:
                dists.append(1.0)
            else:
                centre_fraction = (
                    origin_offset + np.clip(hit_fraction, 0.0, 1.0) * cast_length
                ) / self.OBSTACLE_SENSOR_RANGE
                dists.append(np.clip(centre_fraction, 0.0, 1.0))
        return np.array(dists, dtype=np.float32)

    def _buildRangeDirections(self):
        directions = []
        for elevation_degrees in self.RANGE_ELEVATION_DEGREES:
            elevation = np.deg2rad(elevation_degrees)
            cos_elevation = np.cos(elevation)
            for azimuth_index in range(self.RANGE_AZIMUTH_COUNT):
                azimuth = 2.0 * np.pi * azimuth_index / self.RANGE_AZIMUTH_COUNT
                directions.append(
                    [
                        cos_elevation * np.cos(azimuth),
                        cos_elevation * np.sin(azimuth),
                        np.sin(elevation),
                    ]
                )
        return np.asarray(directions, dtype=np.float32)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        clearance = self._computeNearestTreeClearance(pos)
        collision = self._checkCollision()

        reward, reward_terms, goal_dist = self._reward_model.compute(
            state=state,
            prev_goal_dist=self._prev_goal_dist,
            prev_pos=self._prev_pos,
            start_pos=self.START_POS,
            target_pos=self.TARGET_POS,
            goal_tolerance=self.GOAL_TOLERANCE,
            safe_distance=self.SAFE_DISTANCE,
            clearance=clearance,
            collision=collision,
            terminal_reason=self._classifyDoneReason(state, collision=collision),
        )

        self._last_clearance = clearance
        self._last_collision = collision
        self._prev_goal_dist = goal_dist
        self._prev_pos = np.array(pos, dtype=np.float32)
        self._last_reward_terms = reward_terms
        return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        done_reason = self._classifyDoneReason(state)
        if done_reason in ("success", "collision"):
            self._last_done_reason = done_reason
            return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        done_reason = self._classifyDoneReason(state)
        if done_reason in ("xy_bound", "height_bound", "attitude_bound", "timeout"):
            self._last_done_reason = done_reason
            return True
        return False

    def _classifyDoneReason(self, state, *, collision=None):
        state = np.asarray(state, dtype=np.float32)
        pos = state[0:3]
        goal_dist = float(np.linalg.norm(self.TARGET_POS - pos))
        if goal_dist < self.GOAL_TOLERANCE:
            return "success"
        if self._checkCollision() if collision is None else bool(collision):
            return "collision"
        xy_limit = max(
            self.FOREST_HALF_EXTENT + 0.75,
            float(np.max(np.abs(self.START_POS[:2]))) + 0.5,
            float(np.max(np.abs(self.TARGET_POS[:2]))) + 0.5,
        )

        if self.MAP_SPEC is not None and self.MAP_SPEC.xy_half_extents is not None:
            # New rectangular scenes use their registered domain on each axis.
            # Legacy maps retain the historical margin above.
            outside_xy = np.any(np.abs(pos[:2]) > np.asarray(self.MAP_SPEC.bounds_xy))
        else:
            outside_xy = abs(pos[0]) > xy_limit or abs(pos[1]) > xy_limit
        if outside_xy:
            return "xy_bound"
        if pos[2] < 0.08 or pos[2] > 2.75:
            return "height_bound"
        if abs(state[7]) > 0.65 or abs(state[8]) > 0.65:
            return "attitude_bound"
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return "timeout"
        return "running"

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        goal_dist = float(np.linalg.norm(self.TARGET_POS - pos))
        curriculum_metrics = self.get_curriculum_metrics()
        return {
            "goal_distance": goal_dist,
            "collision": bool(self._last_collision),
            "min_tree_clearance": float(self._last_clearance),
            "success": bool(goal_dist < self.GOAL_TOLERANCE),
            "done_reason": self._last_done_reason,
            "done_reason/success": float(self._last_done_reason == "success"),
            "done_reason/collision": float(self._last_done_reason == "collision"),
            "done_reason/xy_bound": float(self._last_done_reason == "xy_bound"),
            "done_reason/height_bound": float(self._last_done_reason == "height_bound"),
            "done_reason/attitude_bound": float(self._last_done_reason == "attitude_bound"),
            "done_reason/timeout": float(self._last_done_reason == "timeout"),
            **curriculum_metrics,
            "curriculum_episode": int(max(0, self._reset_count - 1)),
            "corridor_half_width": float(self._current_corridor_half_width),
            "corridor_protected": bool(self._protect_corridor),
            "corridor_edge_tree_fraction": float(self._corridor_edge_tree_fraction),
            "centerline_tree_fraction": float(self._centerline_bias_fraction),
            "route_blocking_tree": bool(self.ROUTE_BLOCKING_TREE),
            "route_tree_fraction": float(self.ROUTE_TREE_FRACTION),
            "route_blocking_tree_count": int(self._active_route_blocking_tree_count),
            "route_blocking_tree_count_configured": int(self.ROUTE_BLOCKING_TREE_COUNT),
            "curriculum_primary_route_blocking_tree_count": int(
                self.ROUTE_COUNT_CURRICULUM_COUNTS[self._curriculum_stage]
                if self.ROUTE_COUNT_CURRICULUM
                else self.ROUTE_BLOCKING_TREE_COUNT
            ),
            "curriculum_mixed_difficulty": bool(self._curriculum_mixed_difficulty),
            "range_ray_count": int(self.NUM_RANGE_RAYS),
            "range_ray_origin_offset": float(self.RANGE_RAY_ORIGIN_OFFSET),
            "route_tree_layout": self.ROUTE_TREE_LAYOUT,
            "start_pos": self.START_POS.copy(),
            "target_pos": self.TARGET_POS.copy(),
            "num_trees": int(len(self._tree_specs)),
            "num_obstacles": int(len(self._tree_ids)),
            "map_id": None if self.MAP_SPEC is None else self.MAP_SPEC.map_id,
            "map_source": None if self.MAP_SPEC is None else self.MAP_SPEC.source,
            "map_split": None if self.MAP_SPEC is None else self.MAP_SPEC.split,
            "map_route_reachable": (
                None if self.map_validation is None else self.map_validation.reachable
            ),
            "map_route_validation": (
                None if self.map_validation is None else self.map_validation.message
            ),
            "z_error": float(abs(pos[2] - self.TARGET_POS[2])),
            **{f"reward/{k}": float(v) for k, v in self._last_reward_terms.items()},
        }

    def _computeNearestTreeClearance(self, pos):
        if self._map_runtime is not None:
            return self._map_runtime.nearest_clearance(
                position=pos,
                drone_radius=float(self.COLLISION_R),
                drone_height=float(self.COLLISION_H),
            )
        return self._layout_generator.compute_nearest_tree_clearance(
            pos=pos,
            tree_specs=self._tree_specs,
            drone_radius=float(self.COLLISION_R),
            drone_height=float(self.COLLISION_H),
        )

    def _checkCollision(self):
        contacts = p.getContactPoints(bodyA=int(self.DRONE_IDS[0]), physicsClientId=self.CLIENT)
        obstacle_ids = set(self._tree_ids)
        for contact in contacts:
            other_body = int(contact[2])
            if other_body == int(self.PLANE_ID) or other_body in obstacle_ids:
                return True
        return False

    def set_curriculum_stage_override(self, stage: int | None):
        self._curriculum_stage_override = None if stage is None else int(stage)
        self._applyCurriculum()

    def get_curriculum_stage(self) -> int:
        return int(self._curriculum_stage)

    def get_curriculum_metrics(self) -> dict:
        """Expose scheduler state without allowing evaluation to mutate it."""
        scheduler = self._curriculum_scheduler
        stage = (
            self._curriculum_stage
            if self._curriculum_stage_override is not None or not scheduler.success_gated
            else scheduler.current_stage
        )
        return {
            "curriculum_stage": int(stage),
            "curriculum_success_gated": bool(scheduler.success_gated),
            "curriculum_stage_episode_count": int(scheduler.stage_episode_count),
            "curriculum_recent_episode_count": int(scheduler.recent_episode_count),
            "curriculum_recent_success_rate": float(scheduler.success_rate),
            "curriculum_recent_collision_rate": float(scheduler.collision_rate),
            "curriculum_success_threshold": scheduler.current_success_threshold,
            "curriculum_max_collision_rate": scheduler.current_max_collision_rate,
            "replay_size": int(self._replay_size),
        }

    def set_replay_size(self, replay_size: int):
        self._replay_size = max(0, int(replay_size))

    def record_episode_outcome(
        self,
        success: bool,
        collision: bool = False,
        done_reason: str | None = None,
    ):
        self._curriculum_scheduler.record_episode_outcome(
            success=success,
            collision=collision,
            done_reason=done_reason,
        )

    def _applyCurriculum(self):
        completed_episodes = max(0, self._reset_count - 1)
        stage, config = self._curriculum_scheduler.resolve(
            completed_episodes=completed_episodes,
            override_stage=self._curriculum_stage_override,
        )
        self._curriculum_stage = int(stage)
        self._current_corridor_half_width = float(config.corridor_half_width)
        self._protect_corridor = bool(config.protect_corridor)
        self._corridor_edge_tree_fraction = float(config.corridor_edge_tree_fraction)
        self._centerline_bias_fraction = float(config.centerline_tree_fraction)
        self._active_route_blocking_tree_count = self.ROUTE_BLOCKING_TREE_COUNT
        self._curriculum_mixed_difficulty = False
        if self.ROUTE_COUNT_CURRICULUM:
            primary_count = self.ROUTE_COUNT_CURRICULUM_COUNTS[self._curriculum_stage]
            previous_counts = self.ROUTE_COUNT_CURRICULUM_COUNTS[: self._curriculum_stage]
            if (
                previous_counts
                and self._rng.random() < self.CURRICULUM_SIMPLE_MIX_PROBABILITY
            ):
                self._active_route_blocking_tree_count = int(
                    self._rng.choice(previous_counts)
                )
                self._curriculum_mixed_difficulty = True
            else:
                self._active_route_blocking_tree_count = int(primary_count)
