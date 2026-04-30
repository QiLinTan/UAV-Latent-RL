from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

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
        forest_half_extent: float = 3.0,
        num_trees: int = 24,
        tree_radius_range=(0.10, 0.22),
        tree_height_range=(1.2, 2.4),
        min_tree_separation: float = 0.45,
        corridor_half_width: float = 0.55,
        obstacle_sensor_range: float = 1.75,
        episode_len_sec: float = 12.0,
        safe_distance: float = 0.35,
        goal_tolerance: float = 0.20,
        curriculum: bool = True,
        curriculum_milestones=DEFAULT_CURRICULUM_MILESTONES,
        wide_corridor_half_width: float = 1.35,
        narrow_corridor_half_width: float = 0.35,
        centerline_tree_fraction: float = 0.35,
        centerline_band_width: float = 0.40,
        seed: int | None = None,
        reward_model: ForestRewardModel | None = None,
        curriculum_scheduler: ForestCurriculumScheduler | None = None,
    ):
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
        self.SAFE_DISTANCE = float(safe_distance)
        self.GOAL_TOLERANCE = float(goal_tolerance)
        self.CURRICULUM_ENABLED = bool(curriculum)
        self.CURRICULUM_MILESTONES = tuple(int(x) for x in curriculum_milestones)
        self.WIDE_CORRIDOR_HALF_WIDTH = float(max(wide_corridor_half_width, self.CORRIDOR_HALF_WIDTH))
        self.NARROW_CORRIDOR_HALF_WIDTH = float(min(narrow_corridor_half_width, self.CORRIDOR_HALF_WIDTH))
        self.CENTERLINE_TREE_FRACTION = float(np.clip(centerline_tree_fraction, 0.0, 1.0))
        self.CENTERLINE_BAND_WIDTH = float(centerline_band_width)

        self.NUM_RANGE_RAYS = 8
        self.GOAL_OBS_DIM = 3
        self.OBSTACLE_OBS_DIM = self.GOAL_OBS_DIM + self.NUM_RANGE_RAYS

        self._rng = np.random.default_rng(seed)
        self._tree_ids = []
        self._tree_specs = []
        self._prev_goal_dist = None
        self._prev_pos = self.START_POS.copy()
        self._last_collision = False
        self._last_clearance = np.inf
        self._last_reward_terms = {}
        self._reset_count = 0
        self._curriculum_stage = 0
        self._curriculum_stage_override = None
        self._current_corridor_half_width = self.WIDE_CORRIDOR_HALF_WIDTH
        self._protect_corridor = True
        self._corridor_edge_tree_fraction = 0.0
        self._centerline_bias_fraction = 0.0

        self._layout_generator = ForestLayoutGenerator(
            ForestLayoutConfig(
                forest_half_extent=self.FOREST_HALF_EXTENT,
                num_trees=self.NUM_TREES,
                tree_radius_range=self.TREE_RADIUS_RANGE,
                tree_height_range=self.TREE_HEIGHT_RANGE,
                min_tree_separation=self.MIN_TREE_SEPARATION,
                centerline_band_width=self.CENTERLINE_BAND_WIDTH,
            )
        )
        self._reward_model = reward_model or BaselineForestReward()
        self._curriculum_scheduler = curriculum_scheduler or ForestCurriculumScheduler(
            enabled=self.CURRICULUM_ENABLED,
            milestones=self.CURRICULUM_MILESTONES,
            corridor_half_width=self.CORRIDOR_HALF_WIDTH,
            wide_corridor_half_width=self.WIDE_CORRIDOR_HALF_WIDTH,
            narrow_corridor_half_width=self.NARROW_CORRIDOR_HALF_WIDTH,
            centerline_tree_fraction=self.CENTERLINE_TREE_FRACTION,
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

        info = self._computeInfo()
        return obs, info

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
        tree_specs = self._layout_generator.generate(
            rng=self._rng,
            start_pos=self.INIT_XYZS[0],
            goal_pos=self.TARGET_POS,
            corridor_half_width=self._current_corridor_half_width,
            protect_corridor=self._protect_corridor,
            corridor_edge_tree_fraction=self._corridor_edge_tree_fraction,
            centerline_tree_fraction=self._centerline_bias_fraction,
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
        low = np.hstack([base_space.low, extra_low])
        high = np.hstack([base_space.high, extra_high])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        base_obs = super()._computeObs()
        if self.OBS_TYPE != ObservationType.KIN:
            return base_obs

        state = self._getDroneStateVector(0)
        goal_obs = self._computeGoalObservation(state[0:3]).reshape(1, -1)
        range_obs = self._computeRangeObservation(state[0:3]).reshape(1, -1)
        return np.hstack([base_obs, goal_obs, range_obs]).astype(np.float32)

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

    def _computeRangeObservation(self, pos):
        origin = np.array(pos, dtype=np.float32) + np.array([0.0, 0.0, 0.02], dtype=np.float32)
        dirs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0],
                [0.0, 1.0, 0.0],
                [-0.7071, 0.7071, 0.0],
                [-1.0, 0.0, 0.0],
                [-0.7071, -0.7071, 0.0],
                [0.0, -1.0, 0.0],
                [0.7071, -0.7071, 0.0],
            ],
            dtype=np.float32,
        )
        ray_from = np.repeat(origin.reshape(1, 3), self.NUM_RANGE_RAYS, axis=0)
        ray_to = ray_from + dirs * self.OBSTACLE_SENSOR_RANGE
        hits = p.rayTestBatch(ray_from.tolist(), ray_to.tolist(), physicsClientId=self.CLIENT)

        dists = []
        for hit in hits:
            hit_fraction = float(hit[2])
            if hit_fraction < 0.0:
                dists.append(1.0)
            else:
                dists.append(np.clip(hit_fraction, 0.0, 1.0))
        return np.array(dists, dtype=np.float32)

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
        )

        self._last_clearance = clearance
        self._last_collision = collision
        self._prev_goal_dist = goal_dist
        self._prev_pos = np.array(pos, dtype=np.float32)
        self._last_reward_terms = reward_terms
        return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        goal_dist = np.linalg.norm(self.TARGET_POS - state[0:3])

        if goal_dist < self.GOAL_TOLERANCE:
            return True
        if self._checkCollision():
            return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        xy_limit = max(
            self.FOREST_HALF_EXTENT + 0.75,
            float(np.max(np.abs(self.START_POS[:2]))) + 0.5,
            float(np.max(np.abs(self.TARGET_POS[:2]))) + 0.5,
        )

        if abs(pos[0]) > xy_limit or abs(pos[1]) > xy_limit:
            return True
        if pos[2] < 0.08 or pos[2] > 2.75:
            return True
        if abs(state[7]) > 0.65 or abs(state[8]) > 0.65:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        goal_dist = float(np.linalg.norm(self.TARGET_POS - pos))
        return {
            "goal_distance": goal_dist,
            "collision": bool(self._last_collision),
            "min_tree_clearance": float(self._last_clearance),
            "success": bool(goal_dist < self.GOAL_TOLERANCE),
            "curriculum_stage": int(self._curriculum_stage),
            "curriculum_episode": int(max(0, self._reset_count - 1)),
            "corridor_half_width": float(self._current_corridor_half_width),
            "corridor_protected": bool(self._protect_corridor),
            "corridor_edge_tree_fraction": float(self._corridor_edge_tree_fraction),
            "centerline_tree_fraction": float(self._centerline_bias_fraction),
            "start_pos": self.START_POS.copy(),
            "target_pos": self.TARGET_POS.copy(),
            "num_trees": int(len(self._tree_specs)),
            "z_error": float(abs(pos[2] - self.TARGET_POS[2])),
            **{f"reward/{k}": float(v) for k, v in self._last_reward_terms.items()},
        }

    def _computeNearestTreeClearance(self, pos):
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
