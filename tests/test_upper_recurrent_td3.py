import tempfile
import unittest

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer


class UpperRecurrentTD3Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base_state_dim = 29
        self.history_length = 8
        self.history_frame_dim = 12
        self.state_dim = self.base_state_dim + self.history_length * self.history_frame_dim

    def make_agent(self):
        return TD3UpperRecurrent(
            state_dim=self.state_dim,
            action_dim=3,
            max_action=1.0,
            base_state_dim=self.base_state_dim,
            history_length=self.history_length,
            history_frame_dim=self.history_frame_dim,
            hidden_dim=32,
            device=torch.device("cpu"),
        )

    def test_actor_uses_history_and_preserves_velocity_action_contract(self):
        agent = self.make_agent()
        state_a = np.zeros(self.state_dim, dtype=np.float32)
        state_b = state_a.copy()
        state_b[self.base_state_dim :] = 1.0

        action_a = agent.select_action(state_a)
        action_b = agent.select_action(state_b)

        self.assertEqual(action_a.shape, (3,))
        self.assertLessEqual(float(np.max(np.abs(action_a))), 1.0)
        self.assertFalse(np.allclose(action_a, action_b, atol=1e-7))

    def test_twin_critics_have_independent_recurrent_parameters(self):
        agent = self.make_agent()
        q1_weight = agent.critic.q1.encoder.gru.weight_ih_l0
        q2_weight = agent.critic.q2.encoder.gru.weight_ih_l0
        self.assertNotEqual(q1_weight.data_ptr(), q2_weight.data_ptr())
        self.assertFalse(torch.equal(q1_weight, q2_weight))

    def test_critic_action_scan_returns_one_value_per_candidate(self):
        agent = self.make_agent()
        state = np.zeros(self.state_dim, dtype=np.float32)
        actions = np.array(
            [[0.2, -0.1, z] for z in (-1.0, -0.5, 0.0, 0.5, 1.0)],
            dtype=np.float32,
        )
        q1, q2 = agent.evaluate_critic_actions(state, actions)
        self.assertEqual(q1.shape, (5,))
        self.assertEqual(q2.shape, (5,))
        self.assertTrue(np.all(np.isfinite(q1)))
        self.assertTrue(np.all(np.isfinite(q2)))

    def test_projected_planar_action_matches_executable_unit_disk(self):
        agent = TD3UpperRecurrent(
            state_dim=self.state_dim,
            action_dim=3,
            max_action=1.0,
            base_state_dim=self.base_state_dim,
            history_length=self.history_length,
            history_frame_dim=self.history_frame_dim,
            hidden_dim=32,
            project_planar_action=True,
            device=torch.device("cpu"),
        )
        projected = agent.project_action(np.array([1.0, 1.0, -0.4], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(projected[:2])), 1.0, places=6)
        self.assertAlmostEqual(float(projected[2]), -0.4, places=6)

        states = torch.zeros((8, self.state_dim), dtype=torch.float32)
        actions = agent.actor(states).detach().numpy()
        self.assertTrue(np.all(np.linalg.norm(actions[:, :2], axis=1) <= 1.0 + 1e-6))

    def test_train_and_checkpoint_round_trip(self):
        agent = self.make_agent()
        replay = ReplayBuffer(self.state_dim, 3, max_size=64)
        rng = np.random.default_rng(9)
        for _ in range(32):
            state = rng.normal(size=self.state_dim).astype(np.float32)
            action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
            next_state = rng.normal(size=self.state_dim).astype(np.float32)
            replay.push(state, action, 0.1, next_state, False)

        first = agent.train(replay, batch_size=16)
        second = agent.train(replay, batch_size=16)
        self.assertFalse(first["actor_updated"])
        self.assertTrue(second["actor_updated"])
        self.assertTrue(np.isfinite(second["critic_loss"]))
        self.assertTrue(np.isfinite(second["q_gap_abs_mean"]))

        probe = rng.normal(size=self.state_dim).astype(np.float32)
        expected = agent.select_action(probe)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/upper_recurrent"
            agent.save(prefix)
            restored = self.make_agent()
            restored.load(prefix)
            np.testing.assert_allclose(restored.select_action(probe), expected, atol=1e-7)

    def test_rejects_mismatched_observation_contract(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            TD3UpperRecurrent(
                state_dim=self.state_dim - 1,
                action_dim=3,
                max_action=1.0,
                base_state_dim=self.base_state_dim,
                history_length=self.history_length,
                history_frame_dim=self.history_frame_dim,
                device=torch.device("cpu"),
            )

    def test_success_replay_anchor_and_conservative_phase(self):
        agent = TD3UpperRecurrent(
            state_dim=self.state_dim,
            action_dim=3,
            max_action=1.0,
            base_state_dim=self.base_state_dim,
            history_length=self.history_length,
            history_frame_dim=self.history_frame_dim,
            hidden_dim=32,
            actor_lr=3e-4,
            critic_lr=3e-4,
            conservative_actor_lr=1e-4,
            conservative_critic_lr=1e-4,
            conservative_policy_noise=0.1,
            conservative_policy_freq=3,
            policy_anchor_weight=0.1,
            device=torch.device("cpu"),
        )
        replay = ReplayBuffer(self.state_dim, 3, max_size=64)
        successful_replay = ReplayBuffer(self.state_dim, 3, max_size=64)
        rng = np.random.default_rng(11)
        for _ in range(32):
            transition = (
                rng.normal(size=self.state_dim).astype(np.float32),
                rng.uniform(-1.0, 1.0, size=3).astype(np.float32),
                rng.normal(size=self.state_dim).astype(np.float32),
            )
            replay.push(transition[0], transition[1], 0.1, transition[2], False)
            successful_replay.push(transition[0], transition[1], 0.1, transition[2], False)

        agent.set_policy_anchor_from_current()
        for _ in range(4):
            info = agent.train(
                replay,
                batch_size=16,
                success_replay_buffer=successful_replay,
                success_replay_fraction=0.5,
            )
        self.assertGreater(info["success_replay_fraction"], 0.0)
        self.assertGreaterEqual(info["actor_policy_anchor_loss"], 0.0)

        agent.enter_conservative_phase()
        self.assertTrue(agent.conservative_phase)
        self.assertEqual(agent.policy_freq, 3)
        self.assertAlmostEqual(agent.policy_noise, 0.1)
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()
