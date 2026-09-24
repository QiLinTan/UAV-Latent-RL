import unittest

import numpy as np

from envs.forest.rewards import TerminalGuidanceRateIntegratedReward
from scripts.run_change_td3 import build_mode_cmd


class TerminalGuidanceRewardTest(unittest.TestCase):
    @staticmethod
    def _model(dt=1.0 / 15.0):
        return TerminalGuidanceRateIntegratedReward(
            integration_dt=dt,
            time_penalty_weight=0.0,
            proximity_penalty_weight=0.0,
            near_goal_speed_penalty_weight=0.0,
            near_goal_away_penalty_weight=0.0,
            route_overshoot_penalty_weight=0.0,
            near_goal_guidance_radius=1.2,
            near_goal_guidance_ramp_width=0.2,
            near_goal_capture_speed=0.1,
            near_goal_cruise_speed=0.8,
            near_goal_velocity_tracking_weight=160.0,
            success_terminal_reward=250.0,
            collision_terminal_penalty=300.0,
            bound_terminal_penalty=250.0,
            timeout_terminal_penalty=180.0,
        )

    @staticmethod
    def _compute(model, *, position, velocity, reason="running"):
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = np.asarray(position, dtype=np.float32)
        state[10:13] = np.asarray(velocity, dtype=np.float32)
        target = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        distance = float(np.linalg.norm(target - state[0:3]))
        return model.compute(
            state=state,
            prev_goal_dist=distance,
            prev_pos=state[0:3].copy(),
            start_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            target_pos=target,
            goal_tolerance=0.2,
            safe_distance=0.35,
            clearance=1.0,
            collision=reason == "collision",
            terminal_reason=reason,
        )

    def test_directed_target_velocity_beats_stop_tangent_fast_and_away(self):
        model = self._model()
        # At d=0.5 the linear target speed is 0.31 m/s toward +x.
        desired_reward, desired_terms, _ = self._compute(
            model, position=[0.5, 0.0, 1.0], velocity=[0.31, 0.0, 0.0]
        )
        alternatives = (
            [0.0, 0.0, 0.0],
            [0.0, 0.31, 0.0],
            [0.8, 0.0, 0.0],
            [-0.31, 0.0, 0.0],
        )

        self.assertAlmostEqual(
            desired_terms["near_goal_desired_speed"], 0.31, places=6
        )
        self.assertAlmostEqual(
            desired_terms["near_goal_velocity_tracking_penalty"], 0.0, places=6
        )
        for velocity in alternatives:
            reward, terms, _ = self._compute(
                model, position=[0.5, 0.0, 1.0], velocity=velocity
            )
            self.assertGreater(
                terms["near_goal_velocity_tracking_penalty"], 0.0
            )
            self.assertGreater(desired_reward, reward)

    def test_guidance_is_inactive_outside_terminal_region(self):
        _, terms, _ = self._compute(
            self._model(), position=[-0.5, 0.0, 1.0], velocity=[0.8, 0.0, 0.0]
        )
        self.assertEqual(terms["near_goal_velocity_tracking_penalty"], 0.0)
        self.assertEqual(terms["near_goal_desired_speed"], 0.0)

    def test_target_speed_beats_full_speed_after_progress_is_counted(self):
        model = self._model()
        target = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        start = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        dt = 1.0 / 15.0

        def reward_at(distance, speed):
            position = target - np.array([distance, 0.0, 0.0], dtype=np.float32)
            velocity = np.array([speed, 0.0, 0.0], dtype=np.float32)
            previous_position = position - velocity * dt
            state = np.zeros(16, dtype=np.float32)
            state[0:3] = position
            state[10:13] = velocity
            reward, _, _ = model.compute(
                state=state,
                prev_goal_dist=float(np.linalg.norm(target - previous_position)),
                prev_pos=previous_position,
                start_pos=start,
                target_pos=target,
                goal_tolerance=0.2,
                safe_distance=0.35,
                clearance=1.0,
                collision=False,
                terminal_reason="running",
            )
            return reward

        for distance in (0.3, 0.5, 0.8, 1.0):
            ratio = (distance - 0.2) / (1.2 - 0.2)
            desired_speed = 0.1 + ratio * (0.8 - 0.1)
            with self.subTest(distance=distance):
                self.assertGreater(
                    reward_at(distance, desired_speed), reward_at(distance, 0.8)
                )

    def test_tracking_cost_is_frequency_invariant(self):
        reward_15, terms_15, _ = self._compute(
            self._model(1.0 / 15.0),
            position=[0.5, 0.0, 1.0],
            velocity=[0.0, 0.31, 0.0],
        )
        reward_120, terms_120, _ = self._compute(
            self._model(1.0 / 120.0),
            position=[0.5, 0.0, 1.0],
            velocity=[0.0, 0.31, 0.0],
        )
        self.assertAlmostEqual(15.0 * reward_15, 120.0 * reward_120, places=5)
        self.assertAlmostEqual(
            15.0 * terms_15["near_goal_velocity_tracking_penalty"],
            120.0 * terms_120["near_goal_velocity_tracking_penalty"],
            places=5,
        )

    def test_terminal_outcome_contract_is_unchanged(self):
        reward, terms, _ = self._compute(
            self._model(),
            position=[0.9, 0.0, 1.0],
            velocity=[0.8, 0.0, 0.0],
            reason="success",
        )
        self.assertEqual(reward, 250.0)
        self.assertEqual(terms["terminal_outcome_reward"], 250.0)
        self.assertEqual(terms["near_goal_velocity_tracking_penalty"], 0.0)


class TerminalGuidancePresetTest(unittest.TestCase):
    def test_v2_is_a_controlled_terminal_v1_reward_ablation(self):
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_terminal_v2_h8_k1"
        )

        def value(option):
            return cmd[cmd.index(option) + 1]

        self.assertEqual(value("--reward_variant"), "terminal_guidance_v5")
        self.assertEqual(value("--upper_execution_model"), "k1")
        self.assertEqual(value("--upper_project_planar_action"), "true")
        self.assertEqual(value("--semantic_history_length"), "8")
        self.assertEqual(value("--near_goal_speed_penalty_weight"), "0.0")
        self.assertEqual(value("--near_goal_away_penalty_weight"), "0.0")
        self.assertEqual(value("--near_goal_velocity_tracking_weight"), "160.0")
        self.assertEqual(
            value("--log_dir"),
            "runs/upper_dslpid_recurrent_multitree_terminal_v2_h8_k1",
        )


if __name__ == "__main__":
    unittest.main()
