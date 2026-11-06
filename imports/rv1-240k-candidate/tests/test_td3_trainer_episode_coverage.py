"""Training diagnostics must retain in-episode entry even after leaving the goal."""
from types import SimpleNamespace

import numpy as np

from trainers.td3_trainer import TD3Trainer


class Space:
    def __init__(self, dim):
        self.shape = (1, dim)
        self.high = np.ones((1, dim), dtype=np.float32)


class FakeEnv:
    def __init__(self):
        self.observation_space = Space(12)
        self.action_space = Space(3)
        self.index = 0

    def reset(self, seed=None):
        self.index = 0
        return np.zeros((1, 12), dtype=np.float32), {'goal_distance': 2.0}

    def step(self, action):
        self.index += 1
        distance = (1.1, .6, 1.5)[self.index - 1]
        reward = (2.0, -1.0, -3.0)[self.index - 1]
        done = self.index == 3
        return (np.zeros((1, 12), dtype=np.float32), reward, done, False,
                {'goal_distance': distance, 'done_reason': 'timeout' if done else 'running',
                 'success': False, 'collision': False,
                 'reward/progress_reward': reward if not done else 0.,
                 'reward/continuous_penalty_total': 0. if not done else 0.,
                 'reward/terminal_outcome_reward': -3. if done else 0.,
                 'reward/interval_total': reward})


class FakeAgent:
    def select_action(self, state):
        return np.zeros(3, dtype=np.float32)


def test_training_episode_logs_minimum_distance_and_reward_sums():
    args = SimpleNamespace(seed=0, buffer_size=16, action_scale=1.0,
        success_replay_fraction=0., expl_noise_start=0., expl_noise_end=0.,
        noise_decay_steps=1, start_timesteps=0, reward_scale=.1)
    trainer = TD3Trainer(FakeEnv(), FakeAgent(), args)
    for step in range(1, 4):
        trainer.total_steps = step
        trainer.step_env()
    info = trainer.last_info
    assert info['goal_distance'] == 1.5
    assert info['training_episode/initial_goal_distance'] == 2.0
    assert info['training_episode/min_goal_distance'] == .6
    assert info['training_episode/entered_guidance_1p2'] == 1.
    assert info['training_episode/reward_sum/progress_reward'] == 1.
    assert info['training_episode/reward_sum/terminal_outcome_reward'] == -3.
    assert info['training_episode/reward_sum/interval_total'] == -2.
    assert trainer._episode_min_goal_distance == 2.0
    np.testing.assert_allclose(trainer.buffer.reward[:3, 0], [.2, -.1, -.3])
