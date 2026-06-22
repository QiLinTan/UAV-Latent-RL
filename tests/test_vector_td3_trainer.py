from types import SimpleNamespace

import numpy as np

from algos.td3.td3_plain import TD3Plain
from trainers.vector_td3_trainer import VectorTD3Trainer


class Space:
    def __init__(self, shape):
        self.shape = shape


class FakeVectorEnv:
    autoreset = True

    def __init__(self):
        self.num_envs = 2
        self.observation_space = Space((3,))
        self.action_space = Space((2,))
        self.count = 0

    def reset(self, seed=None):
        return np.zeros((2, 3), dtype=np.float32), [{}, {}]

    def sample_actions(self):
        return np.zeros((2, 2), dtype=np.float32)

    def step(self, actions):
        self.count += 1
        observation = np.full((2, 3), self.count, dtype=np.float32)
        reward = np.ones(2, dtype=np.float32)
        terminated = np.array([self.count % 2 == 0, False])
        truncated = np.array([False, self.count % 3 == 0])
        return observation, reward, terminated, truncated, [{}, {}]


class FakeAgent:
    def __init__(self):
        self.train_calls = 0

    def select_actions(self, states):
        return np.zeros((states.shape[0], 2), dtype=np.float32)

    def train(self, replay_buffer, batch_size):
        assert replay_buffer.size >= batch_size
        self.train_calls += 1
        return {"critic_loss": 1.0}


def test_vector_trainer_collects_batches_and_tracks_update_ratio():
    args = SimpleNamespace(
        seed=1,
        buffer_size=32,
        expl_noise_start=0.0,
        expl_noise_end=0.0,
        noise_decay_steps=1,
        updates_per_transition=0.5,
        start_timesteps=0,
        reward_scale=1.0,
        update_after=0,
        batch_size=2,
        train_every=1,
        total_steps=8,
        progress_interval=100,
    )
    agent = FakeAgent()
    trainer = VectorTD3Trainer(FakeVectorEnv(), agent, args)
    trainer.run()

    assert trainer.total_steps == 8
    assert trainer.buffer.size == 8
    assert agent.train_calls == 4


def test_vector_trainer_runs_one_real_td3_update_on_cpu():
    args = SimpleNamespace(
        seed=1,
        buffer_size=32,
        expl_noise_start=0.0,
        expl_noise_end=0.0,
        noise_decay_steps=1,
        updates_per_transition=0.25,
        start_timesteps=0,
        reward_scale=1.0,
        update_after=0,
        batch_size=2,
        train_every=1,
        total_steps=4,
        progress_interval=100,
    )
    agent = TD3Plain(
        state_dim=3,
        action_dim=2,
        max_action=1.0,
        device="cpu",
    )
    trainer = VectorTD3Trainer(FakeVectorEnv(), agent, args)
    trainer.run()

    assert trainer.buffer.size == 4
    assert agent.total_it == 1
