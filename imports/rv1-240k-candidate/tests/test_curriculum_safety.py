import unittest

import numpy as np

from envs.forest.curriculum import ForestCurriculumScheduler
from envs.forest.rewards import (
    BaselineForestReward,
    MAVRLSafeSpeedForestReward,
    OutcomeSeparatedForestReward,
    RateIntegratedOutcomeForestReward,
    SafeTerminalForestReward,
)


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
            max_collision_rates=(0.30, 0.35, 0.40),
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

    def test_insufficient_samples_do_not_advance(self):
        scheduler = self._scheduler()
        for _ in range(4):
            scheduler.record_episode_outcome(True, collision=False, done_reason="success")
        self.assertEqual(scheduler.current_stage, 0)
        self.assertEqual(scheduler.stage_episode_count, 4)

    def test_success_threshold_without_collision_gate_does_not_advance(self):
        scheduler = self._scheduler()
        for collision in (True, True, False, False, False):
            scheduler.record_episode_outcome(
                success=True,
                collision=collision,
                done_reason="collision" if collision else "success",
            )
        self.assertEqual(scheduler.success_rate, 1.0)
        self.assertAlmostEqual(scheduler.collision_rate, 0.40)
        self.assertEqual(scheduler.current_stage, 0)

    def test_both_gates_advance_exactly_one_stage(self):
        scheduler = self._scheduler()
        for _ in range(5):
            scheduler.record_episode_outcome(True, collision=False, done_reason="success")
        self.assertEqual(scheduler.current_stage, 1)
        self.assertEqual(scheduler.resolve(completed_episodes=10_000)[0], 1)

    def test_promotion_clears_stage_window(self):
        scheduler = self._scheduler()
        for _ in range(5):
            scheduler.record_episode_outcome(True, collision=False, done_reason="success")
        self.assertEqual(scheduler.stage_episode_count, 0)
        self.assertEqual(scheduler.recent_episode_count, 0)
        self.assertEqual(scheduler.success_rate, 0.0)
        self.assertEqual(scheduler.collision_rate, 0.0)

    def test_success_gated_mode_ignores_fixed_milestones(self):
        scheduler = self._scheduler()
        self.assertEqual(scheduler.resolve(completed_episodes=10_000)[0], 0)
        for _ in range(5):
            scheduler.record_episode_outcome(False, collision=False, done_reason="timeout")
        self.assertEqual(scheduler.resolve(completed_episodes=10_000)[0], 0)

    def test_fixed_milestone_mode_is_unchanged(self):
        scheduler = ForestCurriculumScheduler(
            enabled=True,
            milestones=(10, 20, 30),
            corridor_half_width=0.55,
            wide_corridor_half_width=1.35,
            narrow_corridor_half_width=0.35,
            centerline_tree_fraction=0.35,
            success_gated=False,
            success_window=5,
            success_thresholds=(1.0, 1.0, 1.0),
            max_collision_rates=(0.0, 0.0, 0.0),
            minimum_stage_episodes=5,
        )
        for _ in range(50):
            scheduler.record_episode_outcome(False, collision=True, done_reason="collision")
        self.assertEqual(scheduler.resolve(completed_episodes=9)[0], 0)
        self.assertEqual(scheduler.resolve(completed_episodes=10)[0], 1)
        self.assertEqual(scheduler.resolve(completed_episodes=20)[0], 2)
        self.assertEqual(scheduler.resolve(completed_episodes=30)[0], 3)


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


class SafeTerminalRewardTest(unittest.TestCase):
    @staticmethod
    def _compute(reward_model, *, pos, prev_pos, velocity, prev_goal_dist):
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = pos
        state[10:13] = velocity
        return reward_model.compute(
            state=state,
            prev_goal_dist=prev_goal_dist,
            prev_pos=np.asarray(prev_pos, dtype=np.float32),
            start_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=1.0,
            collision=False,
        )

    def test_route_progress_is_capped_after_goal_plane(self):
        legacy_reward, legacy_terms, _ = self._compute(
            BaselineForestReward(),
            pos=[1.2, 0.0, 1.0],
            prev_pos=[1.1, 0.0, 1.0],
            velocity=[0.5, 0.0, 0.0],
            prev_goal_dist=0.1,
        )
        safe_reward, safe_terms, _ = self._compute(
            SafeTerminalForestReward(),
            pos=[1.2, 0.0, 1.0],
            prev_pos=[1.1, 0.0, 1.0],
            velocity=[0.5, 0.0, 0.0],
            prev_goal_dist=0.1,
        )

        self.assertGreater(legacy_terms["route_progress"], 0.0)
        self.assertEqual(safe_terms["route_progress"], 0.0)
        self.assertGreater(safe_terms["route_overshoot_penalty"], 0.0)
        self.assertLess(safe_reward, legacy_reward)

    def test_near_goal_velocity_penalizes_braking_failure_and_moving_away(self):
        _, terms, _ = self._compute(
            SafeTerminalForestReward(),
            pos=[0.5, 0.0, 1.0],
            prev_pos=[0.45, 0.0, 1.0],
            velocity=[-0.6, 0.0, 0.0],
            prev_goal_dist=0.45,
        )

        self.assertGreater(terms["near_goal_speed_penalty"], 0.0)
        self.assertGreater(terms["near_goal_away_penalty"], 0.0)


class MAVRLSafeSpeedRewardTest(unittest.TestCase):
    @staticmethod
    def _compute(reward_model, *, clearance):
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 1.0]
        state[10:13] = [0.6, 0.0, 0.0]
        return reward_model.compute(
            state=state,
            prev_goal_dist=1.0,
            prev_pos=np.zeros(3, dtype=np.float32),
            start_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=clearance,
            collision=False,
        )

    def test_only_speed_above_clearance_conditioned_limit_is_penalized(self):
        reward_model = MAVRLSafeSpeedForestReward(
            safe_speed_min=0.0,
            safe_speed_max=0.8,
            safe_speed_brake_clearance=0.35,
            safe_speed_open_clearance=1.20,
            unsafe_speed_penalty_weight=3.0,
        )
        close_reward, close_terms, _ = self._compute(reward_model, clearance=0.35)
        open_reward, open_terms, _ = self._compute(reward_model, clearance=1.20)

        self.assertAlmostEqual(close_terms["safe_speed"], 0.0, places=6)
        self.assertAlmostEqual(open_terms["safe_speed"], 0.8, places=6)
        self.assertGreater(close_terms["unsafe_speed_penalty"], 0.0)
        self.assertEqual(open_terms["unsafe_speed_penalty"], 0.0)
        self.assertEqual(close_terms["speed_penalty"], 0.0)
        self.assertGreater(open_reward, close_reward)


class OutcomeSeparatedRewardTest(unittest.TestCase):
    @staticmethod
    def _compute(reason):
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = [0.9, 0.0, 1.0]
        return OutcomeSeparatedForestReward().compute(
            state=state,
            prev_goal_dist=0.2,
            prev_pos=np.array([0.8, 0.0, 1.0], dtype=np.float32),
            start_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=1.0,
            collision=reason == "collision",
            terminal_reason=reason,
        )

    def test_absolute_distance_reward_is_removed(self):
        _, terms, _ = self._compute("running")
        self.assertEqual(terms["distance_reward"], 0.0)
        self.assertEqual(terms["terminal_outcome_reward"], 0.0)

    def test_terminal_outcomes_replace_shaping(self):
        success_reward, success_terms, _ = self._compute("success")
        collision_reward, collision_terms, _ = self._compute("collision")
        self.assertEqual(success_reward, 150.0)
        self.assertEqual(success_terms["terminal_outcome_reward"], 150.0)
        self.assertEqual(collision_reward, -120.0)
        self.assertEqual(collision_terms["terminal_outcome_reward"], -120.0)


class RateIntegratedOutcomeRewardTest(unittest.TestCase):
    @staticmethod
    def _compute(model, *, reason="running", clearance=0.10):
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = [0.5, 0.5, 1.0]
        goal_distance = float(
            np.linalg.norm(np.array([1.0, 0.0, 1.0]) - state[0:3])
        )
        return model.compute(
            state=state,
            prev_goal_dist=goal_distance,
            prev_pos=state[0:3].copy(),
            start_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=clearance,
            collision=reason == "collision",
            terminal_reason=reason,
        )

    @staticmethod
    def _model(dt):
        return RateIntegratedOutcomeForestReward(
            integration_dt=dt,
            time_penalty_weight=0.25,
            proximity_penalty_weight=30.0,
            lateral_penalty_scale=0.0,
            success_terminal_reward=250.0,
            collision_terminal_penalty=300.0,
            bound_terminal_penalty=250.0,
            timeout_terminal_penalty=180.0,
        )

    def test_continuous_cost_is_frequency_invariant(self):
        reward_120hz, terms_120hz, _ = self._compute(self._model(1.0 / 120.0))
        reward_15hz, terms_15hz, _ = self._compute(self._model(1.0 / 15.0))

        self.assertAlmostEqual(120.0 * reward_120hz, 15.0 * reward_15hz, places=4)
        self.assertAlmostEqual(
            120.0 * terms_120hz["continuous_penalty_total"],
            15.0 * terms_15hz["continuous_penalty_total"],
            places=5,
        )

    def test_safe_detour_has_no_straight_line_penalty(self):
        _, terms, _ = self._compute(self._model(1.0 / 120.0), clearance=1.0)
        self.assertGreater(terms["route_lateral_offset"], 0.0)
        self.assertEqual(terms["lateral_penalty"], 0.0)

    def test_terminal_outcomes_have_explicit_ordering(self):
        model = self._model(1.0 / 120.0)
        success, _, _ = self._compute(model, reason="success")
        timeout, _, _ = self._compute(model, reason="timeout")
        collision, _, _ = self._compute(model, reason="collision")

        self.assertEqual(success, 250.0)
        self.assertEqual(timeout, -180.0)
        self.assertEqual(collision, -300.0)
        self.assertGreater(success, timeout)
        self.assertGreater(timeout, collision)


if __name__ == "__main__":
    unittest.main()
