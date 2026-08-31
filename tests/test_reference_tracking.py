import unittest

import numpy as np
import torch

from algos.td3.networks import Actor
from algos.td3.td3_reference_tracking import TD3ReferenceTracking
from data.hierarchical_replay_buffer import HierarchicalReplayBuffer
from models.async_reference_channel import AsyncChannelProfile, AsyncReferenceChannel
from models.reference_packet import (
    ActuatorConstraintLayer,
    AsyncReferenceBuffer,
    ReferencePacket,
    RuleReferenceGenerator,
    TrajectoryLimits,
)


def make_packet(version=1, t_gen=0.0, t_receive=0.0):
    return ReferencePacket(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        relative_timestamps=np.array([0.0, 1.0]),
        t_gen=t_gen,
        t_start=t_gen,
        t_receive=t_receive,
        valid_duration=1.0,
        version=version,
        frame_id="local_navigation",
        origin_position=np.array([2.0, 3.0, 1.0], dtype=np.float32),
        origin_attitude=np.zeros(3, dtype=np.float32),
    )


class ReferencePacketTest(unittest.TestCase):
    def test_time_interpolation_and_world_origin(self):
        packet = make_packet()
        sample = packet.sample(0.25, lookahead_seconds=0.5)
        np.testing.assert_allclose(sample["current_position"], [2.25, 3.0, 1.0])
        np.testing.assert_allclose(sample["lookahead_position"], [2.75, 3.0, 1.0])
        self.assertTrue(sample["valid"])

        expired = packet.sample(1.01)
        self.assertFalse(expired["valid"])

    def test_buffer_rejects_stale_and_out_of_order_packets(self):
        buffer = AsyncReferenceBuffer()
        self.assertTrue(buffer.publish(make_packet(version=2, t_gen=1.0, t_receive=1.2)))
        self.assertFalse(buffer.publish(make_packet(version=1, t_gen=1.1, t_receive=1.3)))
        self.assertFalse(buffer.publish(make_packet(version=3, t_gen=0.9, t_receive=1.1)))
        self.assertEqual(buffer.version, 2)
        self.assertEqual(buffer.rejected_packets, 2)
        self.assertEqual(buffer.accepted_packets, 1)
        self.assertEqual(buffer.rejected_stale_version, 1)
        self.assertEqual(buffer.rejected_out_of_order, 1)

    def test_async_channel_replays_latency_drop_duplicate_and_reorder(self):
        profile = AsyncChannelProfile(
            name="fault_test",
            upper_frequency_hz=10.0,
            fixed_latency_seconds=0.1,
            drop_probability=0.0,
            burst_drop_windows=((0.2, 0.3),),
            duplicate_every_n=4,
            reorder_every_n=4,
            reorder_extra_delay_seconds=0.25,
            seed=7,
        )
        channel = AsyncReferenceChannel(profile)
        buffer = AsyncReferenceBuffer()
        self.assertTrue(
            channel.submit(make_packet(version=1, t_gen=0.0, t_receive=0.0), 0.0)
        )
        self.assertTrue(
            channel.submit(make_packet(version=2, t_gen=0.1, t_receive=0.1), 0.1)
        )
        self.assertFalse(
            channel.submit(make_packet(version=3, t_gen=0.2, t_receive=0.2), 0.2)
        )
        self.assertTrue(
            channel.submit(make_packet(version=4, t_gen=0.3, t_receive=0.3), 0.3)
        )
        self.assertEqual(channel.deliver(0.09, buffer), [])
        channel.deliver(0.21, buffer)
        self.assertEqual(buffer.version, 2)
        channel.deliver(0.8, buffer)
        self.assertEqual(buffer.version, 4)
        metrics = channel.metrics()
        self.assertEqual(metrics["burst_drops"], 1)
        self.assertEqual(metrics["duplicated_packets"], 1)
        self.assertEqual(metrics["reordered_packets"], 1)
        self.assertGreaterEqual(metrics["rejected_deliveries"], 1)

    def test_rule_generator_obeys_kinematic_limits(self):
        limits = TrajectoryLimits(
            max_speed=0.6,
            max_acceleration=1.5,
            max_vertical_speed=0.4,
        )
        generator = RuleReferenceGenerator(
            sequence_length=15,
            horizon_seconds=1.0,
            limits=limits,
        )
        packet = generator.generate(
            position=[-3.5, 0.0, 1.0],
            velocity=[0.0, 0.0, 0.0],
            target_position=[3.5, 0.0, 2.0],
            t_gen=0.0,
        )
        feasible, metrics = generator.checker.check(packet)
        self.assertTrue(feasible, metrics)
        self.assertEqual(packet.frame_id, "local_navigation")

    def test_hover_generator_returns_toward_fixed_target(self):
        generator = RuleReferenceGenerator(mode="hover")
        packet = generator.generate(
            position=[0.2, 0.0, 1.0],
            velocity=[0.0, 0.0, 0.0],
            target_position=[0.0, 0.0, 1.0],
            t_gen=0.0,
        )
        world_endpoint = packet.to_world(packet.positions[-1])
        self.assertLess(world_endpoint[0], 0.2)


class ReferenceTrackingAgentTest(unittest.TestCase):
    def _make_agent(self):
        return TD3ReferenceTracking(
            state_dim=263,
            action_dim=4,
            max_action=0.75,
            ctrl_freq=120,
            sequence_length=15,
            reference_horizon_seconds=1.0,
            high_level_interval=8,
            target_position=[3.5, 0.0, 1.0],
            device=torch.device("cpu"),
        )

    @staticmethod
    def _state():
        state = np.zeros(263, dtype=np.float32)
        state[0] = -7.0  # normalized x=-3.5
        state[252:255] = [1.0, 0.0, 0.0]
        state[255:263] = 1.0
        return state

    def test_context_uses_structured_reference_without_latent(self):
        agent = self._make_agent()
        context = agent.prepare_runtime_context(self._state())
        self.assertEqual(context.shape, (agent.context_dim,))
        self.assertEqual(agent.context_dim, 46)
        self.assertEqual(agent.reference_buffer.version, 1)
        self.assertEqual(agent.last_runtime_info["async_cache_valid"], 1.0)
        self.assertGreater(np.linalg.norm(context[18:21]), 0.0)

    def test_new_interface_defaults_to_unrestricted_plain_motor_actor(self):
        agent = self._make_agent()
        self.assertIsInstance(agent.actor, Actor)
        self.assertFalse(hasattr(agent.actor, "collective_limit"))
        self.assertFalse(hasattr(agent.actor, "differential_limit"))

    def test_structured_actor_motor_differentials_are_zero_mean(self):
        agent = TD3ReferenceTracking(
            state_dim=263,
            action_dim=4,
            max_action=0.75,
            actor_structure="structured",
            device=torch.device("cpu"),
        )
        actions = agent.actor(torch.randn(64, agent.context_dim))
        differential = actions - actions.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(differential.mean(dim=1), torch.zeros(64), atol=1e-6))

    def test_expired_packet_enters_separate_degraded_controller(self):
        agent = self._make_agent()
        state = self._state()
        agent.prepare_runtime_context(state)
        agent.previous_action[:] = [0.2, 0.1, -0.1, -0.2]
        agent.set_high_level_enabled(False)
        agent.runtime_step = 121
        context = agent.prepare_runtime_context(state)
        action = agent.action_from_context(context)
        np.testing.assert_allclose(action, 0.85 * agent.previous_action)
        self.assertEqual(agent.last_runtime_info["async_degraded_fallback"], 1.0)

    def test_exact_context_replay_and_training(self):
        agent = self._make_agent()
        replay = HierarchicalReplayBuffer(263, 4, agent.context_dim, max_size=128)
        rng = np.random.default_rng(4)
        for _ in range(96):
            state = self._state()
            state[:12] += rng.normal(0.0, 0.05, 12)
            context = agent.prepare_runtime_context(state)
            action = agent.sample_safe_random_action()
            agent.record_executed_action(action)
            agent.advance_runtime_step()
            next_state = state.copy()
            next_context = agent.prepare_runtime_context(next_state)
            reward = agent.compute_training_reward(
                context=context,
                next_context=next_context,
                next_state=next_state,
                environment_reward=0.0,
                done=False,
                info={},
            )
            replay.push(
                state,
                action,
                reward,
                next_state,
                False,
                context=context,
                next_context=next_context,
            )
        info = agent.train(replay, batch_size=32)
        self.assertTrue(np.isfinite(info["critic_loss"]))
        self.assertEqual(info["reference_tracking_mode"], 1.0)

    def test_actuator_layer_preserves_collective_and_limits_difference(self):
        layer = ActuatorConstraintLayer(4, 0.75, 0.6, 0.25)
        action = layer([2.0, -2.0, 1.0, -1.0])
        self.assertTrue(np.all(np.abs(action) <= 0.75 + 1e-6))
        differential = action - action.mean()
        self.assertLessEqual(float(np.abs(differential).max()), 0.75 * 0.25 + 1e-6)

    def test_actuator_layer_can_limit_motor_slew_rate(self):
        layer = ActuatorConstraintLayer(4, 0.75, 0.6, 0.25, max_delta=0.05)
        previous = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
        action = layer([0.7, 0.7, -0.7, -0.7], previous_action=previous)
        self.assertTrue(np.all(np.abs(action - previous) <= 0.05 + 1e-6))

    def test_episode_reset_clears_teacher_pid_state_and_repeats_output(self):
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel

        class TeacherEnvStub:
            DRONE_MODEL = DroneModel.CF2X
            CTRL_TIMESTEP = 1.0 / 120.0
            HOVER_RPM = 14468.429183500699
            MAX_RPM = 21702.643775480116
            KF = 3.16e-10

            def __init__(self):
                self.motor_action_codec = None
                self.state = np.zeros(20, dtype=np.float64)
                self.state[2] = 1.0
                self.state[6] = 1.0

            def _getDroneStateVector(self, _):
                return self.state.copy()

            def set_motor_action_codec(self, codec):
                self.motor_action_codec = codec

        env = TeacherEnvStub()
        agent = TD3ReferenceTracking(
            state_dim=263,
            action_dim=4,
            max_action=1.0,
            reference_mode="hover",
            device=torch.device("cpu"),
        )
        sample = {
            "lookahead_position": np.array([0.0, 0.0, 1.0]),
            "lookahead_velocity": np.zeros(3),
        }
        agent.last_reference_sample = sample
        first_action = agent.teacher_action_from_env(env)

        controller = agent._teacher_controller
        controller.integral_pos_e[:] = [1.0, -1.0, 0.1]
        controller.integral_rpy_e[:] = [0.5, -0.5, 10.0]
        controller.last_rpy[:] = [0.2, -0.1, 0.3]
        controller.last_rpy_e[:] = [0.1, 0.2, -0.3]
        controller.control_counter = 99

        agent.reset_episode()
        np.testing.assert_allclose(controller.integral_pos_e, 0.0)
        np.testing.assert_allclose(controller.integral_rpy_e, 0.0)
        np.testing.assert_allclose(controller.last_rpy, 0.0)
        np.testing.assert_allclose(controller.last_rpy_e, 0.0)
        self.assertEqual(controller.control_counter, 0)

        agent.last_reference_sample = sample
        second_action = agent.teacher_action_from_env(env)
        np.testing.assert_allclose(second_action, first_action, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
