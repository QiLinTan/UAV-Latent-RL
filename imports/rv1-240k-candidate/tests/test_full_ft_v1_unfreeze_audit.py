"""Observational audit must not perturb training RNG or invent Q evidence."""
import random

import numpy as np
import torch

from scripts.reproduce_full_ft_v1_unfreeze import preserve_rng
from scripts.audit_full_ft_v1_unfreeze import summarize_calibration, td_statistics


def test_observer_preserves_all_cpu_training_rng_streams():
    np.random.seed(43); random.seed(43); torch.manual_seed(43)
    expected = (np.random.random(5), [random.random() for _ in range(5)], torch.rand(5))
    np.random.seed(43); random.seed(43); torch.manual_seed(43)
    with preserve_rng():
        np.random.random(25); random.random(); torch.rand(25)
    actual = (np.random.random(5), [random.random() for _ in range(5)], torch.rand(5))
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_critic_ranking_excludes_actual_return_ties_and_keeps_twin_errors_separate():
    def row(case, name, q, g):
        return {'case_id': case, 'candidate': name, 'initial_q1_q2_min': q,
                'discounted_scaled_return': g, 'outcome': 'success'}
    rows = []
    for case, delta in (('different', -1.), ('tie', .01)):
        rows.append(row(case, 'source', [2., 2., 2.], 10.))
        for name in ('first_update', 'tenth_update', 'step5000_action', 'q1_best_grid', 'minq_best_grid'):
            rows.append(row(case, name, [3., 1., 1.], 10. + delta))
    d = summarize_calibration(rows)['by_candidate']['first_update']
    assert d['pairs'] == 2 and d['distinguishable_pairs'] == 1
    assert d['ranking_error_counts_q1_q2_min'] == [1, 0, 0]


def test_td_probe_uses_saved_target_network_terminal_mask_and_scaled_reward_once():
    class Agent:
        policy_noise = 0.; noise_clip = .5; discount = .99
        def actor_target(self, state):
            return torch.zeros((len(state), 3))
        def critic_target(self, state, action):
            return torch.full((len(state), 1), 10.), torch.full((len(state), 1), 8.)
        def critic(self, state, action):
            return torch.tensor([[1.], [8.92]]), torch.tensor([[1.], [8.92]])
    sample = {'state': torch.zeros((2, 917)), 'next_state': torch.zeros((2, 917)),
              'action': torch.zeros((2, 3)), 'reward': torch.ones((2, 1)),
              'not_done': torch.tensor([[0.], [1.]])}
    result = td_statistics(Agent(), sample, torch.Generator().manual_seed(43))
    assert result['q1']['td_rmse'] < 1e-6
    assert result['q2']['td_rmse'] < 1e-6
