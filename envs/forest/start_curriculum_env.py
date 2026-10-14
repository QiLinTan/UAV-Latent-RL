"""Shared reverse-start/obstacle curricula for K1 and Full forest training."""

import numpy as np

from envs.forest.start_curriculum import START_DISTANCE_BANDS, StartDistanceCurriculum, path_start_candidates
from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.learned_upper_dslpid import KinematicUpperForestEnv, KinematicUpperEnv, LearnedUpperDSLPIDForestEnv


class StartCurriculumMixin:
    """Relocate after map generation; evaluation uses the ordinary environment."""

    def __init__(self, *, start_curriculum_window=100,
                 start_curriculum_minimum_episodes=100,
                 start_curriculum_success_threshold=0.70,
                 start_curriculum_max_collision_rate=0.20,
                 start_curriculum_bands=START_DISTANCE_BANDS, **kwargs):
        self.start_curriculum = StartDistanceCurriculum(
            bands=start_curriculum_bands,
            window=start_curriculum_window,
            minimum_episodes=start_curriculum_minimum_episodes,
            success_threshold=start_curriculum_success_threshold,
            max_collision_rate=start_curriculum_max_collision_rate,
        )
        self._start_rng = np.random.default_rng()
        self._start_source = "original"
        self._start_level = 0
        self._episode_start_position = None
        super().__init__(**kwargs)

    def _start_info(self):
        return {
            "start_curriculum_episode_level": int(self._start_level),
            "start_curriculum_source": self._start_source,
            "start_curriculum_is_band": float(self._start_source == "band"),
            "start_curriculum_is_original": float(self._start_source == "original"),
            "start_curriculum_is_fallback": float(self._start_source.endswith("fallback")),
            "start_curriculum_initial_distance": float(
                np.linalg.norm(self.base_env.TARGET_POS - self._episode_start_position)
            ),
        }

    def _start_candidates(self, band):
        return path_start_candidates(self.base_env, band)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._start_rng = np.random.default_rng(seed)
        observation, info = super().reset(seed=seed, options=options)
        self._start_level = self.start_curriculum.level
        self._start_source = "original"
        self._episode_start_position = np.asarray(self.base_env.pos[0], dtype=np.float32).copy()

        if (self._start_level < len(self.start_curriculum.bands)
                and self._start_rng.random() >= self.start_curriculum.original_probability):
            candidates = self._start_candidates(self.start_curriculum.bands[self._start_level])
            if len(candidates):
                chosen = candidates[int(self._start_rng.integers(len(candidates)))]
                if not self._place_kinematic_state(chosen, np.zeros(3, dtype=np.float32)):
                    self._episode_start_position = np.asarray(self.base_env.pos[0], dtype=np.float32).copy()
                    self._start_source = "band"
                    base = self.base_env
                    base._prev_pos = self._episode_start_position.copy()
                    base._prev_goal_dist = float(np.linalg.norm(base.TARGET_POS - base._prev_pos))
                    base._last_collision = False
                    base._last_clearance = base._computeNearestTreeClearance(base._prev_pos)
                    base._last_done_reason = "running"
                    self.reference_position = self._episode_start_position.copy()
                    self.reference_velocity.fill(0.0)
                    self.reference_acceleration.fill(0.0)
                    self.command_velocity.fill(0.0)
                    base_observation = base._computeObs()
                    if self.local_occupancy is not None:
                        self.local_occupancy.reset()
                        self._update_local_occupancy(base_observation)
                    self._semantic_history.clear()
                    if self.semantic_history_length:
                        frame = self._semantic_frame(base_observation)
                        for _ in range(self.semantic_history_length):
                            self._semantic_history.append(frame.copy())
                    observation = self._upper_observation(base_observation)
                    info = {**info, **base._computeInfo()}
                else:
                    # Never allow an unexpectedly colliding reset into replay.
                    self._place_kinematic_state(self._episode_start_position, np.zeros(3, dtype=np.float32))
                    self._start_source = "collision_fallback"
            else:
                self._start_source = "planner_fallback"
        return observation, {**info, **self._start_info(), **self.start_curriculum.metrics()}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, {
            **info, **self._start_info(), **self.start_curriculum.metrics()
        }

    def record_episode_outcome(self, success, collision=False, done_reason=None):
        self.start_curriculum.record(
            source=self._start_source,
            success=success,
            collision=collision or done_reason == "collision",
        )
        # Forest difficulty is deliberately frozen at Stage 0 in this ablation.
        # Do not advance the independent obstacle scheduler behind the override.
        return None

    def get_curriculum_metrics(self):
        return {**super().get_curriculum_metrics(), **self.start_curriculum.metrics()}


class StartObstacleCurriculumMixin(StartCurriculumMixin):
    """Unlock random obstacle stages after the original-start course reaches L4."""

    def __init__(self, *, obstacle_curriculum_window=100,
                 obstacle_curriculum_minimum_episodes=100,
                 obstacle_curriculum_success_thresholds=(0.80, 0.75, 0.70),
                 obstacle_curriculum_max_collision_rates=(0.15, 0.20, 0.20),
                 **kwargs):
        self.obstacle_curriculum = ObstacleCurriculum(
            route_counts=kwargs.get("route_count_curriculum_counts", (1, 2, 3, 5)),
            window=obstacle_curriculum_window,
            minimum_episodes=obstacle_curriculum_minimum_episodes,
            success_thresholds=obstacle_curriculum_success_thresholds,
            max_collision_rates=obstacle_curriculum_max_collision_rates,
        )
        self._obstacle_episode_stage = 0
        self._obstacle_episode_primary = True
        super().__init__(**kwargs)
        if not self.base_env.ROUTE_COUNT_CURRICULUM:
            raise ValueError("Combined curriculum requires route-count curriculum")
        if tuple(self.base_env.ROUTE_COUNT_CURRICULUM_COUNTS) != self.obstacle_curriculum.route_counts:
            raise ValueError("Obstacle scheduler counts differ from map curriculum counts")
        self.base_env.set_curriculum_stage_override(0)

    def _obstacle_episode_info(self):
        return {
            "obstacle_curriculum_episode_stage": self._obstacle_episode_stage,
            "obstacle_curriculum_episode_primary": float(self._obstacle_episode_primary),
            **self.obstacle_curriculum.metrics(),
        }

    def reset(self, *, seed=None, options=None):
        if self.base_env._curriculum_stage_override != self.obstacle_curriculum.stage:
            self.base_env.set_curriculum_stage_override(self.obstacle_curriculum.stage)
        observation, info = super().reset(seed=seed, options=options)
        self._obstacle_episode_stage = int(info["curriculum_stage"])
        self._obstacle_episode_primary = (
            not bool(info["curriculum_mixed_difficulty"])
            and int(info["route_blocking_tree_count"])
            == self.obstacle_curriculum.route_counts[self._obstacle_episode_stage]
        )
        if self._obstacle_episode_stage != self.obstacle_curriculum.stage:
            raise AssertionError("Generated obstacle stage differs from scheduler stage")
        return observation, {**info, **self._obstacle_episode_info()}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, {
            **info, **self._obstacle_episode_info()
        }

    def record_episode_outcome(self, success, collision=False, done_reason=None):
        super().record_episode_outcome(success, collision, done_reason)
        promoted = self.obstacle_curriculum.record(
            episode_stage=self._obstacle_episode_stage,
            start_level=self._start_level,
            primary_difficulty=self._obstacle_episode_primary,
            success=success,
            collision=collision or done_reason == "collision",
        )
        if promoted:
            self.base_env.set_curriculum_stage_override(self.obstacle_curriculum.stage)
        return promoted

    def get_curriculum_metrics(self):
        return {**super().get_curriculum_metrics(), **self._obstacle_episode_info()}


class StartCurriculumKinematicUpperForestEnv(StartCurriculumMixin, KinematicUpperForestEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.kinematic_model != "k1":
            raise ValueError("Start curriculum requires the K1 execution model")


class StartObstacleCurriculumKinematicUpperForestEnv(StartObstacleCurriculumMixin,
                                                    StartCurriculumKinematicUpperForestEnv):
    pass


class StartObstacleCurriculumFullForestEnv(StartObstacleCurriculumMixin, LearnedUpperDSLPIDForestEnv):
    """Same reset-only course; all steps execute the real Full/DSLPID plant."""

    # This helper only places the rigid body at reset and refreshes collision
    # detection. No kinematic integration or K1 step enters the Full MRO.
    _place_kinematic_state = KinematicUpperEnv._place_kinematic_state

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Map-atlas overrides constructor arguments: enforce after construction.
        self.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.0
