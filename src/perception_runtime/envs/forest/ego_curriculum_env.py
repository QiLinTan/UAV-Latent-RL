"""Full multi-map distance and nested density curriculum."""
import copy

import numpy as np

from envs.forest.start_curriculum_env import StartCurriculumMixin
from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv, KinematicUpperEnv
from envs.map_atlas.forest_pool import ForestPool, COUNTS, DEFAULT_MANIFEST

BANDS = ((.35, 1.2), (1.2, 4.), (4., 8.), (8., float('inf')))


class EgoCurriculumFullForestEnv(StartCurriculumMixin, LearnedUpperDSLPIDForestEnv):
    _place_kinematic_state = KinematicUpperEnv._place_kinematic_state

    def __init__(self, *, forest_pool_path=DEFAULT_MANIFEST, pool_split='train',
                 training_course=True, allow_test=False,
                 obstacle_curriculum_window=100, obstacle_curriculum_minimum_episodes=100,
                 obstacle_curriculum_success_thresholds=(.80, .75, .70),
                 obstacle_curriculum_max_collision_rates=(.15, .20, .20), **kwargs):
        if pool_split == 'test' and not allow_test:
            raise ValueError('Sealed test pool requires explicit final evaluation')
        if training_course and pool_split != 'train':
            raise ValueError('Only train geometry may supply training replay')
        self.pool = ForestPool(forest_pool_path)
        self.pool_split = pool_split
        self.training_course = bool(training_course)
        self.obstacle_curriculum = ObstacleCurriculum(
            route_counts=COUNTS, window=obstacle_curriculum_window,
            minimum_episodes=obstacle_curriculum_minimum_episodes,
            success_thresholds=obstacle_curriculum_success_thresholds,
            max_collision_rates=obstacle_curriculum_max_collision_rates)
        self._pool_rng = np.random.default_rng()
        self._pool_draw_count = 0
        self._pool_visits = {}
        self._pool_index = 0
        self._density_episode_stage = 0
        self._remaining_path_length = None
        kwargs['map_spec'] = self.pool.spec(pool_split, 0, 0)
        kwargs['start_curriculum_bands'] = BANDS
        super().__init__(**kwargs)
        self.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.
        if not self.training_course:
            self.start_curriculum.level = len(BANDS)

    def _start_candidates(self, band):
        path = self.pool.path_for(self.pool_split, self._pool_index, self._density_episode_stage)
        remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)[::-1])[::-1], 0.]
        mask = (remaining >= band[0]) & (remaining < band[1])
        mask &= np.linalg.norm(path - self.base_env.TARGET_POS, axis=1) > max(.30, self.base_env.GOAL_TOLERANCE + .10)
        candidates = path[mask]
        candidates = np.asarray([p for p in candidates if self.base_env._computeNearestTreeClearance(p) >= .18], dtype=np.float32)
        if not candidates.size:
            raise RuntimeError(f'No band reset candidates for {self._pool_info()} band={band}')
        return candidates.reshape(-1, 3)

    def _pool_info(self):
        entry = self.pool.entries(self.pool_split)[self._pool_index]
        data = entry['stages'][self._density_episode_stage]
        return dict(map_id=entry['record']['map_id'], map_split=self.pool_split,
                    map_seed=entry['seed'], map_pool_index=self._pool_index,
                    map_pool_fingerprint=self.pool.fingerprint,
                    geometry_sha256=data['geometry_sha256'],
                    map_route_reachable=True, map_route_validation='frozen envelope-inflated BFS QA',
                    current_density_full_path_length=data['path_length'],
                    density_stage=self._density_episode_stage, curriculum_stage=self._density_episode_stage,
                    active_tree_count=COUNTS[self._density_episode_stage], primary_difficulty=True,
                    curriculum_mixed_difficulty=False,
                    start_curriculum_remaining_path_length=self._remaining_path_length)

    def reset(self, *, seed=None, options=None):
        if seed is not None and self.training_course:
            self._pool_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261013]))
        self._density_episode_stage = self.obstacle_curriculum.stage
        if self.training_course:
            self._pool_index = int(self._pool_rng.integers(len(self.pool.entries(self.pool_split))))
        else:
            # Evaluation seed is a fixed pool index, not a map-generating seed.
            self._pool_index = int(seed if seed is not None else self._pool_draw_count) % len(self.pool.entries(self.pool_split))
        self._pool_draw_count += 1
        self._pool_visits[self._pool_index] = self._pool_visits.get(self._pool_index, 0) + 1
        self.base_env.MAP_SPEC = self.pool.spec(self.pool_split, self._pool_index, self._density_episode_stage)
        self.base_env.map_validation = None  # cached, envelope-inflated QA is in the manifest
        observation, info = super().reset(seed=seed, options=options)
        path = self.pool.path_for(self.pool_split, self._pool_index, self._density_episode_stage)
        remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)[::-1])[::-1], 0.]
        node = int(np.argmin(np.linalg.norm(path - self.pos[0], axis=1)))
        self._remaining_path_length = float(remaining[node])
        if len(self.base_env._tree_ids) != COUNTS[self._density_episode_stage]:
            raise AssertionError('Collision body density differs from curriculum')
        return observation, {**info, **self._pool_info(), **self._density_metrics()}

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        return state, reward, terminated, truncated, {**info, **self._pool_info(), **self._density_metrics()}

    def record_episode_outcome(self, success, collision=False, done_reason=None):
        if not self.training_course:
            return False
        # Gate using the level at reset; the last L3 episode cannot certify L4.
        completed_at_reset = self._start_level >= len(self.start_curriculum.bands)
        super().record_episode_outcome(success, collision, done_reason)
        return self.obstacle_curriculum.record(
            episode_stage=self._density_episode_stage, start_level=self._start_level,
            start_complete=completed_at_reset and self._start_source == 'original',
            primary_difficulty=True, success=success, collision=collision or done_reason == 'collision')

    def _density_metrics(self):
        values = self.obstacle_curriculum.metrics()
        values.pop('obstacle_curriculum_primary_route_count')
        return {**values, 'obstacle_curriculum_active_tree_count': COUNTS[self.obstacle_curriculum.stage]}

    def get_curriculum_stage(self):
        return self.obstacle_curriculum.stage

    def set_curriculum_stage_override(self, stage):
        if stage is not None:
            self.obstacle_curriculum.stage = int(stage)

    def get_curriculum_metrics(self):
        return {**self._pool_info(), **self.start_curriculum.metrics(), **self._density_metrics()}

    def sampler_state(self):
        return copy.deepcopy(dict(fingerprint=self.pool.fingerprint, split=self.pool_split,
                                  rng=self._pool_rng.bit_generator.state, draws=self._pool_draw_count,
                                  visits=self._pool_visits, index=self._pool_index))

    def restore_sampler_state(self, state):
        if state['fingerprint'] != self.pool.fingerprint or state['split'] != self.pool_split:
            raise ValueError('Cannot resume with changed pool or split')
        self._pool_rng.bit_generator.state = state['rng']
        self._pool_draw_count = state['draws']
        self._pool_visits = copy.deepcopy(state['visits'])
        self._pool_index = state['index']
