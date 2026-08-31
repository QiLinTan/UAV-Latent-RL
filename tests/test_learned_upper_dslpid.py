import unittest

import numpy as np

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from envs.learned_upper_dslpid import LearnedUpperDSLPIDEnv


class LearnedUpperDSLPIDEnvTest(unittest.TestCase):
    def setUp(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
        )
        self.env = LearnedUpperDSLPIDEnv(
            base_env,
            upper_control_interval=8,
        )

    def tearDown(self):
        self.env.close()

    def test_upper_step_executes_reference_through_frozen_dslpid(self):
        observation, reset_info = self.env.reset(seed=7)
        self.assertEqual(observation.shape, (1, 29))
        self.assertEqual(self.env.action_space.shape, (1, 3))
        self.assertEqual(reset_info["controller_mode"], "learned_upper_dslpid")

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )

        self.assertEqual(next_observation.shape, (1, 29))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["lower_steps_executed"], 8)
        self.assertTrue(info["reference_packet_valid"])
        self.assertEqual(self.env.runtime_lower_step, 8)
        self.assertGreater(info["reference_endpoint_x"], float(self.env.pos[0][0]))
        self.assertTrue(np.all(np.isfinite(self.env.last_motor_action)))
        self.assertLessEqual(float(np.max(np.abs(self.env.last_motor_action))), 1.0)
        self.assertAlmostEqual(info["velocity_command_x"], 0.6, places=5)
        self.assertLessEqual(
            info["reference_acceleration_norm"],
            self.env.max_reference_acceleration + 0.01,
        )

    def test_replanning_preserves_reference_position_and_velocity(self):
        self.env.reset(seed=11)
        self.env.step(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        previous_position = self.env.reference_position.copy()
        previous_velocity = self.env.reference_velocity.copy()

        self.env.step(np.array([[-1.0, 0.0, 0.0]], dtype=np.float32))
        packet = self.env.last_reference_packet

        np.testing.assert_allclose(packet.positions[0], previous_position, atol=1e-6)
        np.testing.assert_allclose(packet.velocities[0], previous_velocity, atol=1e-6)
        self.assertLessEqual(
            float(np.max(np.abs(self.env.reference_acceleration))),
            self.env.max_reference_acceleration + 1e-5,
        )

    def test_semantic_history_extends_observation_without_changing_action_interface(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
        )
        env = LearnedUpperDSLPIDEnv(
            base_env,
            upper_control_interval=8,
            semantic_history_length=4,
        )
        try:
            observation, info = env.reset(seed=3)
            self.assertEqual(observation.shape, (1, 77))
            self.assertEqual(info["semantic_history_length"], 4)
            next_observation, *_ = env.step(
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (1, 77))
            self.assertEqual(env.action_space.shape, (1, 3))
            self.assertFalse(np.array_equal(observation[:, 29:], next_observation[:, 29:]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
