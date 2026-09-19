import unittest
from unittest import mock

import numpy as np
import torch

from algos.td3.td3_reference_tracking import (
    STAGE2_V1_CONTEXT_ID,
    TD3ReferenceTracking,
    stage2_v1_context_contract,
)
from models.continuous_world_reference import (
    ContinuousWorldPacketSource,
    TrapezoidalWorldTrajectory,
)
from models.motor_action_codec import ASYMMETRIC_RPM, ASYMMETRIC_THRUST
from scripts import validate_stage1_reference_pid, validate_stage2_reference_tracking


class Stage2ExternalReferenceTrackingTest(unittest.TestCase):
    @staticmethod
    def make_agent(**overrides):
        kwargs = dict(
            state_dim=263,
            action_dim=4,
            max_action=1.0,
            target_position=[3.5, 0.0, 1.0],
            ctrl_freq=120,
            sequence_length=16,
            reference_horizon_seconds=1.0,
            high_level_interval=8,
            reference_mode="line",
            reference_source="external_packet",
            max_reference_speed=0.6,
            max_reference_acceleration=1.5,
            max_reference_vertical_speed=0.4,
            lookahead_points=3,
            lower_action_history_steps=0,
            motor_action_mode=ASYMMETRIC_RPM,
            device=torch.device("cpu"),
        )
        kwargs.update(overrides)
        return TD3ReferenceTracking(**kwargs)

    @staticmethod
    def make_source():
        trajectory = TrapezoidalWorldTrajectory(
            [-3.5, 0.0, 1.0],
            [3.5, 0.0, 1.0],
            max_speed=0.6,
            max_acceleration=1.5,
        )
        return trajectory, ContinuousWorldPacketSource(
            trajectory,
            window_seconds=1.0,
            sequence_length=16,
        )

    @staticmethod
    def normalized_initial_state():
        state = np.zeros(263, dtype=np.float32)
        state[0] = -7.0  # world x=-3.5 normalized by 0.5
        return state

    def test_external_mode_never_calls_legacy_generator(self):
        agent = self.make_agent()
        self.assertIsNone(agent.reference_generator)
        agent._refresh_reference = mock.Mock(
            side_effect=AssertionError("legacy generator must not run")
        )
        context = agent.prepare_runtime_context(self.normalized_initial_state())
        agent._refresh_reference.assert_not_called()
        self.assertEqual(context.shape, (30,))
        self.assertEqual(agent.last_runtime_info["async_degraded_fallback"], 1.0)

    def test_stage1_and_stage2_import_the_same_packet_source(self):
        self.assertIs(
            validate_stage1_reference_pid.ContinuousWorldPacketSource,
            ContinuousWorldPacketSource,
        )
        self.assertIs(
            validate_stage2_reference_tracking.ContinuousWorldPacketSource,
            ContinuousWorldPacketSource,
        )

    def test_external_time_sampling_separates_current_and_future_reference(self):
        agent = self.make_agent()
        trajectory, source = self.make_source()
        self.assertTrue(agent.publish_reference(source.make_packet(0.0)))
        state = self.normalized_initial_state()
        context = agent.prepare_runtime_context(state)

        control_time = 1.0 / 120.0
        expected = agent.reference_buffer.sample(
            control_time,
            lookahead_seconds=0.2,
        )
        sample = agent.last_reference_sample
        self.assertAlmostEqual(sample["execution_time"], control_time)
        np.testing.assert_allclose(sample["current_position"], expected["current_position"])
        np.testing.assert_allclose(sample["current_velocity"], expected["current_velocity"])
        np.testing.assert_allclose(sample["lookahead_position"], expected["lookahead_position"])
        np.testing.assert_allclose(sample["lookahead_velocity"], expected["lookahead_velocity"])
        self.assertFalse(np.array_equal(context[12:15], context[18:21]))

    def test_stage2_v1_context_contract_is_exactly_30d_without_history(self):
        agent = self.make_agent()
        contract = agent.context_contract()
        self.assertEqual(contract, stage2_v1_context_contract())
        self.assertEqual(len(contract), 30)
        self.assertEqual([field["index"] for field in contract], list(range(30)))
        self.assertEqual(agent.lower_action_history_dim, 0)
        self.assertEqual(contract[12]["name"], "current_position_error_x")
        self.assertEqual(contract[18]["name"], "future_position_error_x")
        self.assertEqual(contract[24]["name"], "reference_age_ratio")
        self.assertEqual(contract[26]["name"], "previous_motor_action_0")

    def test_external_mode_locks_asymmetric_rpm_and_no_action_history(self):
        with self.assertRaisesRegex(ValueError, "asymmetric_rpm"):
            self.make_agent(motor_action_mode=ASYMMETRIC_THRUST)
        with self.assertRaisesRegex(ValueError, "lower_action_history_steps=0"):
            self.make_agent(lower_action_history_steps=4)

    def test_validator_rejects_checkpoint_without_frozen_external_contract(self):
        agent = self.make_agent()
        checkpoint = {
            "actor_state_dict": agent.actor.state_dict(),
            "architecture": {
                "input_dim": 30,
                "output_dim": 4,
                "multi_step_action_history": 0,
            },
            "motor_action_codec": ASYMMETRIC_RPM,
            "context_contract_id": STAGE2_V1_CONTEXT_ID,
        }
        with mock.patch.object(
            validate_stage2_reference_tracking.torch,
            "load",
            return_value=checkpoint,
        ):
            validate_stage2_reference_tracking.load_actor_checkpoint(agent, "unused.pt")

        incompatible = dict(checkpoint, context_contract_id="legacy_30d")
        with mock.patch.object(
            validate_stage2_reference_tracking.torch,
            "load",
            return_value=incompatible,
        ):
            with self.assertRaisesRegex(ValueError, "frozen Stage2-v1"):
                validate_stage2_reference_tracking.load_actor_checkpoint(
                    agent,
                    "unused.pt",
                )

    def test_expired_external_packet_has_explicit_previous_action_decay_fallback(self):
        agent = self.make_agent()
        _, source = self.make_source()
        self.assertTrue(agent.publish_reference(source.make_packet(0.0)))
        agent.previous_action[:] = [0.2, 0.1, -0.1, -0.2]
        agent.runtime_step = 121
        context = agent.prepare_runtime_context(self.normalized_initial_state())
        action = agent.action_from_context(context)

        self.assertEqual(agent.last_runtime_info["async_cache_valid"], 0.0)
        self.assertEqual(agent.last_runtime_info["async_degraded_fallback"], 1.0)
        np.testing.assert_allclose(
            action,
            0.85 * np.array([0.2, 0.1, -0.1, -0.2], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
