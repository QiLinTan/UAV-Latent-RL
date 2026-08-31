import unittest

import numpy as np

from envs.forest.curriculum import ForestCurriculumScheduler
from envs.forest.rewards import BaselineForestReward


class PerformanceCurriculumTest(unittest.TestCase):
    def _scheduler(self):
        return ForestCurriculumScheduler(
            enabled=True,
            milestones=(10, 20, 30),
            corridor_half_width=0.55,
            wide_corridor_half_width=1.35,
            narrow_corridor_half_width=0.35,
            centerline_tree_fraction=0.35,
            success_gated=True,
            success_window=5,
            success_thresholds=(0.60, 0.70, 0.80),
            minimum_stage_episodes=5,
        )

    def test_short_failed_episodes_do_not_advance_curriculum(self):
        scheduler = self._scheduler()
        for _ in range(50):
            scheduler.record_episode_outcome(False)
        stage, _ = scheduler.resolve(completed_episodes=50)
        self.assertEqual(stage, 0)

    def test_success_rate_advances_one_stage(self):
        scheduler = self._scheduler()
        for outcome in (True, True, True, False, False):
            scheduler.record_episode_outcome(outcome)
        stage, _ = scheduler.resolve(completed_episodes=5)
        self.assertEqual(stage, 1)


class SafetyBoundaryRewardTest(unittest.TestCase):
    def test_attitude_near_termination_receives_large_penalty(self):
        reward_model = BaselineForestReward(safety_boundary_penalty_weight=25.0)
        state = np.zeros(16, dtype=np.float32)
        state[2] = 1.0
        state[7] = 0.64
        reward, terms, _ = reward_model.compute(
            state=state,
            prev_goal_dist=1.0,
            prev_pos=np.zeros(3, dtype=np.float32),
            start_pos=np.zeros(3, dtype=np.float32),
            target_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=1.0,
            collision=False,
        )
        self.assertGreater(terms["safety_boundary_penalty"], 15.0)
        self.assertLess(reward, -10.0)


if __name__ == "__main__":
    unittest.main()
