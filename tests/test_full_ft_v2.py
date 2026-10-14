"""Fail-closed state comparison and predeclared ablation decisions."""
import copy
from collections import deque

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer
from scripts.run_full_ft_v2 import differences, fully_collapsed, stable_for_extension, json_values
from scripts.reproduce_full_ft_v1_unfreeze import preserve_rng


def evaluation(stage0, full3, k1):
    return {name: {'summary': {'eval/success_rate': value}}
            for name, value in zip(('full_stage0', 'full_stage3', 'k1_stage3'), (stage0, full3, k1))}


def test_undefined_tracking_metrics_are_null_without_masking_success_rates():
    value = json_values({'summary': {'success': .8, 'tracking': float('nan')}, 'finite': False})
    assert value == {'summary': {'success': .8, 'tracking': None}, 'finite': False}


def test_prefix_comparator_catches_rng_replay_optimizer_and_control_differences():
    state = {'tensor': torch.zeros(2), 'history': deque([np.array([1., 2.])]),
             'rng': np.random.default_rng(43), 'optimizer': {'state': {1: {'step': torch.tensor(2.)}}}}
    assert not differences(state, copy.deepcopy(state))
    other = copy.deepcopy(state)
    other['rng'].random()
    other['history'][0][1] = 3.
    other['optimizer']['state'][1]['step'] += 1
    paths = differences(state, other)
    assert any('rng' in p for p in paths)
    assert any('history' in p for p in paths)
    assert any('optimizer' in p for p in paths)


def test_stop_and_extension_gates_require_all_three_domains():
    assert fully_collapsed(evaluation(0., 0., 0.))
    assert not fully_collapsed(evaluation(0., 0., .1))
    assert stable_for_extension(evaluation(.8, .7, .8))
    assert not stable_for_extension(evaluation(.7, .8, 1.))
    assert not stable_for_extension(evaluation(1., .6, 1.))
    assert not stable_for_extension(evaluation(1., 1., .7))


def test_source_action_anchor_has_negligible_effect_on_first_actor_update():
    torch.manual_seed(18)
    plain = TD3UpperRecurrent(11, 3, 1., base_state_dim=5, history_length=2,
                             history_frame_dim=3, hidden_dim=8, device=torch.device('cpu'))
    anchored = copy.deepcopy(plain)
    anchored.policy_anchor_weight = .1
    anchored.set_policy_anchor_from_current()
    source = copy.deepcopy(anchored._policy_anchor_actor.state_dict())
    replay = ReplayBuffer(11, 3, max_size=32)
    rng = np.random.default_rng(18)
    for _ in range(32):
        replay.push(rng.normal(size=11), rng.uniform(-.5, .5, size=3), 1., rng.normal(size=11), False)
    for _ in range(2):
        with preserve_rng():
            left = plain.train(replay, batch_size=16, success_replay_buffer=replay, success_replay_fraction=.2)
        right = anchored.train(replay, batch_size=16, success_replay_buffer=replay, success_replay_fraction=.2)
    assert left['actor_updated'] and right['actor_updated']
    # Grad-enabled and no-grad GRU kernels can differ at floating-point epsilon.
    assert right['actor_policy_anchor_loss'] < 1e-12
    for key, value in plain.actor.state_dict().items():
        torch.testing.assert_close(value, anchored.actor.state_dict()[key], rtol=0., atol=1e-7)
    assert not differences(anchored._policy_anchor_actor.state_dict(), source)
    assert all(p.grad is None and not p.requires_grad for p in anchored._policy_anchor_actor.parameters())
