import os
import tempfile
import unittest

import numpy as np
import torch

from algos.td3.td3_hierarchical_async import TD3HierarchicalAsync
from data.hierarchical_replay_buffer import HierarchicalReplayBuffer
from models.hierarchical_async import AsyncReferenceCache, ReferenceSequenceHead


class AsyncReferenceCacheTest(unittest.TestCase):
    def test_interpolates_and_expires_cached_sequence(self):
        cache = AsyncReferenceCache()
        sequence = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        cache.update(
            sequence,
            np.array([0.25, -0.5], dtype=np.float32),
            created_step=10,
            point_interval=4,
            valid_steps=12,
        )

        sample = cache.sample(12, lookahead_points=2)
        np.testing.assert_allclose(sample["current"], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(sample["lookahead"], [2.0, 0.0, 0.0])
        self.assertTrue(sample["valid"])
        self.assertEqual(sample["version"], 1)

        expired = cache.sample(23)
        self.assertFalse(expired["valid"])
        self.assertEqual(expired["age_steps"], 13)

    def test_reference_head_has_sequence_shape(self):
        head = ReferenceSequenceHead(latent_dim=16, sequence_length=15)
        actual = head(torch.zeros(4, 16))
        self.assertEqual(actual.shape, (4, 15, 3))
        self.assertTrue(torch.all(actual.abs() <= 1.0))


class HierarchicalAsyncTD3Test(unittest.TestCase):
    def _make_agent(self):
        return TD3HierarchicalAsync(
            state_dim=263,
            action_dim=4,
            max_action=1.0,
            latent_dim=8,
            sequence_length=6,
            high_level_interval=4,
            reference_valid_steps=16,
            device=torch.device("cpu"),
        )

    def test_runtime_refreshes_at_high_level_interval(self):
        agent = self._make_agent()
        state = np.zeros(263, dtype=np.float32)
        state[252] = 1.0
        state[255:263] = 1.0

        for _ in range(5):
            action = agent.select_action(state)
            self.assertEqual(action.shape, (4,))
            self.assertTrue(np.isfinite(action).all())

        self.assertEqual(agent.reference_cache.version, 2)
        self.assertEqual(agent.last_runtime_info["async_high_level_refreshed"], 1.0)

    def test_expired_cache_uses_hover_equivalent_fallback(self):
        agent = self._make_agent()
        state = np.zeros(263, dtype=np.float32)
        state[252] = 1.0
        state[255:263] = 1.0
        agent.select_action(state)
        agent.set_high_level_enabled(False)
        agent.runtime_step = agent.reference_valid_steps + 1

        action = agent.select_action(state)

        np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))
        self.assertEqual(agent.last_runtime_info["async_cache_valid"], 0.0)
        self.assertEqual(agent.last_runtime_info["async_safe_fallback"], 1.0)

    def test_train_save_and_load(self):
        agent = self._make_agent()
        replay = HierarchicalReplayBuffer(263, 4, agent.context_dim, max_size=128)
        rng = np.random.default_rng(7)
        for _ in range(96):
            state = rng.normal(0.0, 0.2, size=263).astype(np.float32)
            next_state = rng.normal(0.0, 0.2, size=263).astype(np.float32)
            state[252:255] = [1.0, 0.0, 0.0]
            next_state[252:255] = [0.98, 0.0, 0.0]
            state[255:263] = rng.uniform(0.2, 1.0, size=8)
            next_state[255:263] = rng.uniform(0.2, 1.0, size=8)
            context = rng.normal(0.0, 0.2, size=agent.context_dim).astype(np.float32)
            next_context = rng.normal(0.0, 0.2, size=agent.context_dim).astype(np.float32)
            replay.push(
                state,
                rng.uniform(-1.0, 1.0, size=4),
                0.1,
                next_state,
                False,
                context=context,
                next_context=next_context,
            )

        info = agent.train(replay, batch_size=32)
        self.assertTrue(np.isfinite(info["critic_loss"]))
        self.assertTrue(np.isfinite(info["reference_loss"]))
        self.assertEqual(info["hierarchical_async"], 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            agent.save(path)
            restored = self._make_agent()
            restored.load(path)
            action = restored.select_action(replay.state[0])
            self.assertTrue(np.isfinite(action).all())

    def test_context_replay_preserves_exact_async_context(self):
        agent = self._make_agent()
        replay = HierarchicalReplayBuffer(263, 4, agent.context_dim, max_size=8)
        state = np.zeros(263, dtype=np.float32)
        state[252] = 1.0
        state[255:263] = 1.0
        context = agent.prepare_runtime_context(state)
        action = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
        agent.record_executed_action(action)
        agent.advance_runtime_step()
        next_context = agent.prepare_runtime_context(state)
        replay.push(
            state,
            action,
            0.2,
            state,
            False,
            context=context,
            next_context=next_context,
        )

        sample = replay.sample(1)
        np.testing.assert_allclose(sample[5].numpy()[0], context)
        np.testing.assert_allclose(sample[6].numpy()[0], next_context)
        np.testing.assert_allclose(
            next_context[-4:],
            action,
        )

    def test_motor_actor_limits_differential_commands(self):
        agent = self._make_agent()
        context = torch.randn(128, agent.context_dim)
        actions = agent.actor(context)
        differential = actions - actions.mean(dim=1, keepdim=True)
        self.assertTrue(torch.all(actions.abs() <= agent.max_action + 1e-6))
        self.assertLessEqual(
            float(differential.abs().max().item()),
            2.0 * agent.max_action * agent.motor_differential_fraction + 1e-6,
        )

    def test_random_and_noisy_exploration_respect_motor_projection(self):
        agent = self._make_agent()
        for _ in range(100):
            random_action = agent.sample_safe_random_action()
            noisy_action = agent.add_safe_exploration_noise(random_action, noise_std=0.5)
            for action in (random_action, noisy_action):
                self.assertTrue(np.all(np.abs(action) <= agent.max_action + 1e-6))
                differential = action - action.mean()
                self.assertLessEqual(
                    float(np.abs(differential).max()),
                    agent.max_action * agent.motor_differential_fraction + 1e-6,
                )


if __name__ == "__main__":
    unittest.main()
