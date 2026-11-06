import json
from pathlib import Path
import unittest

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel

from algos.td3.td3_reference_tracking import TD3ReferenceTracking
from models.canonical_dslpid_teacher import (
    CANONICAL_DSLPID_WORLD_RATE_ID,
    CanonicalDSLPIDTeacher,
    WORLD_ANGULAR_VELOCITY,
    previous_rpy_from_world_angular_velocity,
)
from models.continuous_world_reference import ContinuousWorldPacketSource
from scripts import (
    analyze_stage2_v1_label_ambiguity,
    collect_stage2_v1_dataset,
    validate_stage1_reference_pid,
    validate_stage2_reference_tracking,
    validate_stage2_v1_teacher_determinization,
)


class Stage2V1DatasetContractTest(unittest.TestCase):
    def test_collector_uses_the_frozen_shared_packet_source(self):
        self.assertIs(
            collect_stage2_v1_dataset.ContinuousWorldPacketSource,
            ContinuousWorldPacketSource,
        )
        self.assertIs(
            validate_stage1_reference_pid.ContinuousWorldPacketSource,
            ContinuousWorldPacketSource,
        )
        self.assertIs(
            validate_stage2_reference_tracking.ContinuousWorldPacketSource,
            ContinuousWorldPacketSource,
        )

    def test_pilot_configuration_freezes_stage2_v1_time_contract(self):
        config = collect_stage2_v1_dataset.Stage2PilotConfig()
        self.assertEqual(config.packet_frequency_hz, 15)
        self.assertEqual(config.control_frequency_hz, 120)
        self.assertEqual(config.packet_interval_steps, 8)
        self.assertEqual(config.sequence_length, 16)
        self.assertEqual(config.lookahead_points, 3)
        self.assertAlmostEqual(config.lookahead_seconds, 0.2)

    def test_pilot_seed_profiles_are_explicit_and_bounded(self):
        profiles = [
            collect_stage2_v1_dataset.disturbance_for_seed(seed)
            for seed in (0, 1, 2)
        ]
        self.assertEqual(profiles[0].name, "nominal")
        for profile in profiles:
            self.assertLessEqual(max(np.abs(profile.initial_rpy)), 0.04)
            self.assertLessEqual(max(np.abs(profile.initial_velocity)), 0.12)
            self.assertLessEqual(max(np.abs(profile.initial_angular_velocity)), 0.12)
            self.assertLessEqual(abs(profile.height_offset), 0.04)
        with self.assertRaises(ValueError):
            collect_stage2_v1_dataset.disturbance_for_seed(3)

    def test_formal_seed_profiles_are_paired_bounded_and_split_disjoint(self):
        seeds = list(collect_stage2_v1_dataset.FORMAL_SEEDS)
        self.assertEqual(seeds, list(range(51)))
        nominal = collect_stage2_v1_dataset.formal_disturbance_for_seed(0)
        self.assertEqual(nominal.name, "nominal")
        for positive_seed in range(1, 51, 2):
            positive = collect_stage2_v1_dataset.formal_disturbance_for_seed(
                positive_seed
            )
            negative = collect_stage2_v1_dataset.formal_disturbance_for_seed(
                positive_seed + 1
            )
            np.testing.assert_allclose(positive.initial_rpy, -np.asarray(negative.initial_rpy))
            np.testing.assert_allclose(
                positive.initial_velocity,
                -np.asarray(negative.initial_velocity),
            )
            np.testing.assert_allclose(
                positive.initial_angular_velocity,
                -np.asarray(negative.initial_angular_velocity),
            )
            self.assertAlmostEqual(positive.height_offset, -negative.height_offset)
            self.assertLessEqual(max(np.abs(positive.initial_rpy)), 0.04)
            self.assertLessEqual(max(np.abs(positive.initial_velocity)), 0.12)
            self.assertLessEqual(
                max(np.abs(positive.initial_angular_velocity)),
                0.12,
            )
            self.assertLessEqual(abs(positive.height_offset), 0.04)
        splits = {
            name: {
                seed
                for seed in seeds
                if collect_stage2_v1_dataset.formal_split_for_seed(seed) == name
            }
            for name in ("train", "validation", "test")
        }
        self.assertEqual([len(splits[name]) for name in splits], [39, 6, 6])
        self.assertFalse(splits["train"] & splits["validation"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["validation"] & splits["test"])
        self.assertEqual(set.union(*splits.values()), set(seeds))

    def test_context_reconstruction_matches_frozen_field_order(self):
        state = np.array(
            [-3.5, 0.1, 1.2, 0.03, -0.02, 0.01, 0.4, -0.2, 0.1, 0.8, -0.4, 0.2],
            dtype=np.float32,
        )
        current = np.array([-3.4, 0.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32)
        future = np.array([-3.3, 0.0, 1.0, 0.6, 0.0, 0.0], dtype=np.float32)
        previous = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        context = collect_stage2_v1_dataset.reconstruct_context(
            state,
            current,
            future,
            0.25,
            True,
            previous,
        )
        self.assertEqual(context.shape, (30,))
        np.testing.assert_allclose(
            context[:12],
            (state - TD3ReferenceTracking._KIN_MEAN) / TD3ReferenceTracking._KIN_STD,
        )
        np.testing.assert_allclose(context[26:30], previous)
        self.assertAlmostEqual(float(context[24]), 0.25)
        self.assertEqual(float(context[25]), 1.0)

    def test_dataset_schema_rejects_any_sample_axis_misalignment(self):
        valid = {
            "context": np.zeros((3, 30), dtype=np.float32),
            "state": np.zeros((3, 12), dtype=np.float32),
            "state_quaternion": np.zeros((3, 4), dtype=np.float32),
        }
        self.assertEqual(
            collect_stage2_v1_dataset.validate_sample_axes(valid),
            3,
        )
        invalid = dict(valid, state_quaternion=np.zeros((6, 4), dtype=np.float32))
        with self.assertRaisesRegex(AssertionError, "state_quaternion"):
            collect_stage2_v1_dataset.validate_sample_axes(invalid)

    @staticmethod
    def synthetic_ambiguity_inputs(action_gap: float):
        samples_per_episode = 120
        base = np.linspace(-0.02, 0.02, samples_per_episode, dtype=np.float64)
        first = np.zeros((samples_per_episode, 30), dtype=np.float64)
        first[:, 0] = base
        second = first.copy()
        contexts = np.concatenate([first, second])
        actions = np.zeros((2 * samples_per_episode, 4), dtype=np.float64)
        actions[samples_per_episode:] = action_gap
        wrenches = np.zeros((2 * samples_per_episode, 4), dtype=np.float64)
        wrenches[:, 1:] = 1.0
        if action_gap:
            wrenches[samples_per_episode:, 1:] = -1.0
        episode_ids = np.repeat([0, 1], samples_per_episode)
        hidden = np.zeros((2 * samples_per_episode, 13), dtype=np.float64)
        return contexts, actions, wrenches, episode_ids, hidden

    def test_local_ambiguity_gate_accepts_consistent_cross_episode_labels(self):
        metrics = analyze_stage2_v1_label_ambiguity.local_ambiguity_metrics(
            *self.synthetic_ambiguity_inputs(action_gap=0.0)
        )
        self.assertEqual(metrics["close_pair_count"], 240)
        self.assertTrue(metrics["gate_passed"])
        json.dumps(metrics)

    def test_local_ambiguity_gate_rejects_same_context_different_labels(self):
        metrics = analyze_stage2_v1_label_ambiguity.local_ambiguity_metrics(
            *self.synthetic_ambiguity_inputs(action_gap=0.2)
        )
        self.assertFalse(metrics["gate_passed"])
        self.assertGreaterEqual(
            metrics["close_pair_action_linf"]["p95"],
            0.2 - 1e-9,
        )
        self.assertGreater(metrics["salient_torque_sign_conflict_rate"], 0.02)

    def test_canonical_teacher_repeats_output_after_unrelated_calls(self):
        teacher = CanonicalDSLPIDTeacher(
            DroneModel.CF2X,
            rate_mode=WORLD_ANGULAR_VELOCITY,
        )
        inputs = {
            "control_timestep": 1.0 / 120.0,
            "cur_pos": np.array([-3.4, 0.05, 1.03]),
            "cur_quat": p.getQuaternionFromEuler([0.03, -0.02, 0.01]),
            "cur_vel": np.array([0.2, -0.1, 0.03]),
            "cur_ang_vel": np.array([0.1, -0.08, 0.04]),
            "target_pos": np.array([-3.3, 0.0, 1.0]),
            "target_vel": np.array([0.3, 0.0, 0.0]),
        }
        first = teacher.computeControl(**inputs)[0]
        teacher.computeControl(
            **dict(
                inputs,
                cur_pos=np.array([-2.0, 0.2, 1.2]),
                target_pos=np.array([-1.8, 0.0, 1.0]),
            )
        )
        repeated = teacher.computeControl(**inputs)[0]
        np.testing.assert_array_equal(first, repeated)
        self.assertIsNotNone(teacher.last_hidden_before)
        self.assertIsNotNone(teacher.last_hidden_after)
        self.assertEqual(teacher.last_hidden_before["control_counter"], 0)
        np.testing.assert_array_equal(
            teacher.last_hidden_before["integral_pos_e"],
            np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(
            CANONICAL_DSLPID_WORLD_RATE_ID,
            "stage2_v1_canonical_dslpid_world_rate",
        )

    def test_canonical_ambiguity_gate_uses_selected_teacher_reproduction(self):
        self.assertIn(
            "canonical_teacher_reproduction_metrics",
            dir(analyze_stage2_v1_label_ambiguity),
        )
        source = Path(analyze_stage2_v1_label_ambiguity.__file__).read_text(
            encoding="utf-8"
        )
        self.assertIn("selected_teacher_label_uniqueness", source)
        self.assertIn("training_allowed", source)

    def test_world_angular_velocity_defines_previous_rpy_observably(self):
        quaternion = p.getQuaternionFromEuler([0.03, -0.02, 0.01])
        current = np.asarray(p.getEulerFromQuaternion(quaternion))
        previous = previous_rpy_from_world_angular_velocity(
            quaternion,
            [0.12, -0.08, 0.05],
            1.0 / 120.0,
        )
        self.assertFalse(np.array_equal(current, previous))
        self.assertLess(np.linalg.norm(current - previous), 0.01)
        stationary = previous_rpy_from_world_angular_velocity(
            quaternion,
            [0.0, 0.0, 0.0],
            1.0 / 120.0,
        )
        np.testing.assert_allclose(stationary, current)

    def test_teacher_probe_requires_all_task_episodes_for_a_mode(self):
        passing = {
            "task": "hover",
            "gate_passed": True,
            "motor_saturation_ratio": 0.0,
            "missing_packet_count": 0,
            "stale_packet_count": 0,
            "fallback_count": 0,
            "packet_version_continuity": True,
        }
        episodes = [
            dict(passing, task="hover"),
            dict(passing, task="straight"),
        ]
        summary = validate_stage2_v1_teacher_determinization.summarize_mode(
            "test",
            episodes,
        )
        self.assertTrue(summary["passed"])
        episodes[1]["gate_passed"] = False
        summary = validate_stage2_v1_teacher_determinization.summarize_mode(
            "test",
            episodes,
        )
        self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
