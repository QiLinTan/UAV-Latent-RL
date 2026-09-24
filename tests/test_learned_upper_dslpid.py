import unittest

import numpy as np
import pybullet as p

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from envs.learned_upper_dslpid import KinematicUpperEnv, LearnedUpperDSLPIDEnv
from models.reference_packet import ReferencePacket


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
        self.assertIsInstance(self.env.last_reference_packet, ReferencePacket)
        self.assertEqual(self.env.last_reference_packet.frame_id, "world")
        self.assertEqual(info["reference_packet_version"], 1)
        self.assertEqual(self.env.runtime_lower_step, 8)
        self.assertGreater(info["reference_endpoint_x"], float(self.env.pos[0][0]))
        self.assertTrue(np.all(np.isfinite(self.env.last_motor_action)))
        self.assertLessEqual(float(np.max(np.abs(self.env.last_motor_action))), 1.0)
        self.assertAlmostEqual(info["reward/interval_total"], reward, places=6)
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

    def test_planar_only_upper_action_uses_deterministic_target_height_hold(self):
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
            planar_only=True,
            height_hold_gain=1.0,
        )
        try:
            _, reset_info = env.reset(seed=5)
            self.assertEqual(env.action_space.shape, (1, 2))
            self.assertTrue(reset_info["planar_only"])
            self.assertAlmostEqual(reset_info["height_hold_target"], 1.0, places=5)

            _, _, terminated, truncated, info = env.step(
                np.array([[1.0, 0.0]], dtype=np.float32)
            )

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["planar_only"])
            self.assertAlmostEqual(info["velocity_command_x"], 0.6, places=5)
            self.assertAlmostEqual(info["velocity_command_z"], 0.0, places=5)
            with self.assertRaises(ValueError):
                env._velocity_command(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        finally:
            env.close()

    def test_96_ray_3d_sensor_preserves_reference_packet_contract(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        env = LearnedUpperDSLPIDEnv(
            base_env,
            upper_control_interval=8,
            semantic_history_length=4,
        )
        try:
            observation, info = env.reset(seed=9)
            self.assertEqual(base_env.NUM_RANGE_RAYS, 96)
            self.assertEqual(base_env._range_directions.shape, (96, 3))
            self.assertEqual(observation.shape, (1, 517))
            self.assertEqual(info["range_ray_count"], 96)
            next_observation, _, _, _, step_info = env.step(
                np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (1, 517))
            self.assertIsInstance(env.last_reference_packet, ReferencePacket)
            self.assertTrue(step_info["reference_packet_valid"])
        finally:
            env.close()

    def test_local_occupancy_is_appended_once_outside_recurrent_history(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        env = KinematicUpperEnv(
            base_env,
            upper_control_interval=8,
            semantic_history_length=8,
            kinematic_model="k1",
            use_local_occupancy=True,
            local_occupancy_voxel_size=0.5,
            local_occupancy_shape=(4, 4, 4),
        )
        try:
            observation, info = env.reset(seed=19)
            self.assertEqual(env.base_upper_obs_dim, 117 + 64)
            self.assertEqual(observation.shape, (1, 917 + 64))
            self.assertTrue(info["local_occupancy_enabled"])
            self.assertEqual(info["local_occupancy_dim"], 64)
            self.assertEqual(
                info["local_occupancy_unknown_voxels"]
                + info["local_occupancy_free_voxels"]
                + info["local_occupancy_occupied_voxels"],
                64,
            )
            occupancy_values = observation[0, 117 : 117 + 64]
            self.assertTrue(set(np.unique(occupancy_values)).issubset({-1.0, 0.0, 1.0}))
            next_observation, _, _, _, step_info = env.step(
                np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, observation.shape)
            self.assertTrue(step_info["local_occupancy_enabled"])
        finally:
            env.close()

    def test_3d_range_sensor_excludes_the_drone_itself(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        try:
            base_env.reset(seed=10)
            ray_from, ray_to, _, _ = base_env._rangeRayGeometry(base_env.pos[0])
            raw_hits = p.rayTestBatch(
                ray_from.tolist(), ray_to.tolist(), physicsClientId=base_env.CLIENT
            )
            self.assertNotIn(int(base_env.DRONE_IDS[0]), [int(hit[0]) for hit in raw_hits])
            ranges = base_env._computeRangeObservation(base_env.pos[0])
            self.assertEqual(ranges.shape, (96,))
            # At z=1 m the 1.75 m range does not reach the floor along a
            # -30 degree ray. With no trees, every ray should be clear. This
            # specifically catches the former self-hit bank at fraction~0.01.
            np.testing.assert_allclose(ranges, np.ones(96, dtype=np.float32), atol=1e-6)
        finally:
            base_env.close()

    def test_3d_range_sensor_detects_ground_without_self_hits(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        try:
            base_env.reset(seed=11)
            low_position = np.array([-3.5, 0.0, 0.4], dtype=np.float32)
            p.resetBasePositionAndOrientation(
                int(base_env.DRONE_IDS[0]),
                low_position.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=base_env.CLIENT,
            )
            base_env._updateAndStoreKinematicInformation()
            ranges = base_env._computeRangeObservation(base_env.pos[0])
            down, horizontal, up = ranges[:32], ranges[32:64], ranges[64:]
            self.assertTrue(np.all(down > 0.1))
            self.assertTrue(np.all(down < 0.8))
            np.testing.assert_allclose(horizontal, np.ones(32), atol=1e-6)
            np.testing.assert_allclose(up, np.ones(32), atol=1e-6)
        finally:
            base_env.close()

    def test_3d_range_sensor_restores_centre_referenced_distance(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        try:
            base_env.reset(seed=12)
            shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                height=0.2,
                physicsClientId=base_env.CLIENT,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=shape,
                basePosition=[-2.5, 0.0, 1.0],
                physicsClientId=base_env.CLIENT,
            )
            ranges = base_env._computeRangeObservation(base_env.pos[0])
            # Horizontal elevation is the second 32-ray bank; azimuth 0 points
            # along +x. The cylinder surface is 0.9 m from the drone centre.
            self.assertAlmostEqual(
                float(ranges[32]),
                0.9 / base_env.OBSTACLE_SENSOR_RANGE,
                delta=0.02,
            )
            np.testing.assert_allclose(ranges[64:], np.ones(32), atol=1e-6)
        finally:
            base_env.close()

    def test_k0_preserves_upper_contract_and_executes_velocity_ideally(self):
        base_env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            num_trees=0,
            route_blocking_tree=False,
            curriculum=False,
            range_azimuth_count=32,
            range_elevation_degrees=(-30.0, 0.0, 30.0),
        )
        env = KinematicUpperEnv(
            base_env,
            upper_control_interval=8,
            semantic_history_length=4,
            kinematic_model="k0",
        )
        try:
            observation, info = env.reset(seed=13)
            self.assertEqual(observation.shape, (1, 517))
            self.assertEqual(info["upper_execution_model"], "k0")
            initial_x = float(env.pos[0][0])
            next_observation, _, terminated, truncated, step_info = env.step(
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (1, 517))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(
                float(env.pos[0][0] - initial_x),
                0.6 * env.CTRL_TIMESTEP,
                places=5,
            )
            self.assertAlmostEqual(float(env.vel[0][0]), 0.6, places=5)
            self.assertAlmostEqual(step_info["reference_position_error"], 0.0, places=6)
            self.assertFalse(step_info["reference_packet_valid"])
            self.assertEqual(step_info["lower_steps_executed"], 0)
        finally:
            env.close()

    def test_k1_enforces_velocity_acceleration_limit(self):
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
        env = KinematicUpperEnv(
            base_env,
            upper_control_interval=8,
            kinematic_model="k1",
            kinematic_planar_acceleration=1.5,
        )
        try:
            env.reset(seed=17)
            _, _, _, _, info = env.step(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
            self.assertAlmostEqual(
                float(env.vel[0][0]),
                1.5 * env.CTRL_TIMESTEP,
                places=5,
            )
            self.assertAlmostEqual(
                info["command_actual_delta_v"],
                0.6 - 1.5 * env.CTRL_TIMESTEP,
                places=5,
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
