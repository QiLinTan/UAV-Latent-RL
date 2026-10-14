import unittest

import numpy as np

from envs.forest.start_curriculum import StartDistanceCurriculum
from envs.forest.start_curriculum_env import StartCurriculumKinematicUpperForestEnv
from scripts.audit_stage0_feasibility import BASE, make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args


class StartCurriculumSchedulerTest(unittest.TestCase):
    def test_only_current_band_episodes_advance(self):
        scheduler = StartDistanceCurriculum(window=3, minimum_episodes=3,
                                            success_threshold=2 / 3,
                                            max_collision_rate=1 / 3)
        for _ in range(10):
            scheduler.record(source="original", success=True, collision=False)
        self.assertEqual(scheduler.level, 0)
        for outcome in ((True, False), (True, False), (False, True)):
            scheduler.record(source="band", success=outcome[0], collision=outcome[1])
        self.assertEqual(scheduler.level, 1)
        self.assertEqual(scheduler.level_episodes, 0)

    def test_failed_band_episodes_hold_level(self):
        scheduler = StartDistanceCurriculum(window=3, minimum_episodes=3)
        for _ in range(10):
            scheduler.record(source="band", success=False, collision=False)
        self.assertEqual(scheduler.level, 0)


class StartCurriculumPhysicalResetTest(unittest.TestCase):
    def test_map_is_unchanged_and_relocated_state_is_consistent(self):
        args = preset_args(BASE)
        original = make_stage0_env(args)
        course = make_stage0_env(args, env_class=StartCurriculumKinematicUpperForestEnv)
        original.set_curriculum_stage_override(0)
        course.set_curriculum_stage_override(0)
        try:
            # This seed selects the short band with the independent reset RNG.
            _, original_info = original.reset(seed=0)
            observation, info = course.reset(seed=0)
            self.assertEqual(info["start_curriculum_source"], "band")
            self.assertEqual(info["curriculum_stage"], 0)
            self.assertEqual(len(original.base_env._tree_specs), len(course.base_env._tree_specs))
            for before, after in zip(original.base_env._tree_specs, course.base_env._tree_specs):
                np.testing.assert_allclose(before["xy"], after["xy"])
                self.assertEqual(before["radius"], after["radius"])
            self.assertAlmostEqual(original_info["goal_distance"], 7.0, places=4)
            self.assertGreater(info["goal_distance"], 0.2)
            self.assertLess(info["goal_distance"], 1.2)
            self.assertFalse(course.base_env._checkCollision())
            pos = course.base_env.pos[0]
            np.testing.assert_allclose(course.base_env._prev_pos, pos)
            np.testing.assert_allclose(course.reference_position, pos)
            np.testing.assert_allclose(course.reference_velocity, np.zeros(3))
            self.assertAlmostEqual(course.base_env._prev_goal_dist, info["goal_distance"], places=5)
            self.assertEqual(observation.shape, course.observation_space.shape)
            self.assertTrue(all(np.allclose(course._semantic_history[0], frame)
                                for frame in course._semantic_history))
            _, _, _, _, step_info = course.step(np.zeros(course.action_space.shape, dtype=np.float32))
            self.assertEqual(step_info["start_curriculum_source"], "band")
            self.assertLess(abs(step_info["reward/progress_reward"]), 0.01)
        finally:
            course.close()
            original.close()
