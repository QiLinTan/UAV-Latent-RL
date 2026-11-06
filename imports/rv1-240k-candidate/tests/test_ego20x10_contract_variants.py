"""Behavioral checks for isolated contract candidates and optimizer rollback."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from envs.forest.rewards import TerminalGuidanceRateIntegratedReward
from scripts.audit_full_ego20x10_task_contract import raw_state, reward_on
from scripts.ego20x10_contract_variants import (
    ActorStepGuard, CONTRACTS, FeasibleGuidanceReward, encode_observation,
    reencode_replay_state, project_velocity, critic_calibration_step,
)


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


def test_goal_range_preserves_16m_distance_and_unchanged_history():
    obs = np.zeros(917, dtype=np.float32)
    obs[:3] = [-8, 0, 1]
    obs[12:15] = [1, 0, 0]
    baseline = encode_observation(obs, CONTRACTS['baseline'])
    converted = reencode_replay_state(np.stack([baseline, baseline]), CONTRACTS['goal_range'])
    assert converted[0, 12] == pytest.approx(.8)
    np.testing.assert_array_equal(converted[:, 117:], np.stack([baseline[117:]]*2))
    baseline[0] = -6/.5
    assert reencode_replay_state(baseline, CONTRACTS['goal_range'])[12] == pytest.approx(.7)
    scaled = encode_observation(obs, CONTRACTS['position_scale'])
    np.testing.assert_allclose(scaled[:3], [-.8, 0, 0])
    np.testing.assert_array_equal(scaled[3:], encode_observation(obs, CONTRACTS['baseline'])[3:])


def test_feasible_guidance_replaces_only_tracking_and_preserves_terminals():
    original = TerminalGuidanceRateIntegratedReward(integration_dt=1/120)
    candidate = FeasibleGuidanceReward(original)
    env = SimpleNamespace(TARGET_POS=np.array([0, 0, 1], dtype=np.float32),
                          START_POS=np.array([-16, 0, 1]), GOAL_TOLERANCE=.2, SAFE_DISTANCE=.35)
    state = raw_state([0, 0, 1.8], [0, 0, -.3])
    old = reward_on(original, env, state, state)
    new = reward_on(candidate, env, state, state)
    assert old[1]['near_goal_velocity_tracking_penalty'] == pytest.approx(7.744/120, rel=1e-5)
    assert new[1]['near_goal_velocity_tracking_penalty'] == pytest.approx(0, abs=1e-12)
    assert new[0]-old[0] == pytest.approx(old[1]['near_goal_velocity_tracking_penalty'])
    for reason in ('success', 'collision', 'bounds', 'timeout'):
        equal_nested(reward_on(original, env, state, state, terminal=reason),
                     reward_on(candidate, env, state, state, terminal=reason))
    feasible = raw_state([-.8, 0, 1], [.2, 0, 0])
    equal_nested(reward_on(original, env, feasible, feasible), reward_on(candidate, env, feasible, feasible))
    v = project_velocity([.8, .8, 2.])
    assert np.linalg.norm(v[:2]) == pytest.approx(.8)
    assert v[2] == pytest.approx(.3)


def agent_and_buffer():
    torch.manual_seed(43)
    agent = TD3UpperRecurrent(917, 3, 1., base_state_dim=117, history_length=8,
        history_frame_dim=100, hidden_dim=64, project_planar_action=True, device=torch.device('cpu'))
    states = torch.randn(8, 917)
    class Buffer:
        size = 8
        def sample(self, count):
            assert count == 8
            return states, torch.zeros(8, 3), states*.9, torch.ones(8, 1), torch.ones(8, 1)
    agent.total_it = 1
    return agent, Buffer(), states


def test_rejected_actor_step_restores_adam_and_target_but_keeps_critic_step():
    torch.set_num_threads(1)
    agent, buffer, states = agent_and_buffer()
    # Exercise rollback with populated Adam, not only a never-used optimizer.
    agent.train(buffer, 8)
    agent.total_it = 3
    actor = copy.deepcopy(agent.actor.state_dict())
    target = copy.deepcopy(agent.actor_target.state_dict())
    adam = copy.deepcopy(agent.actor_optimizer.state_dict())
    critic = copy.deepcopy(agent.critic.state_dict())
    guard = ActorStepGuard(agent, states, mean_limit=0., p95_limit=0., backtracks=0)
    info, report = guard.train(buffer, 8)
    assert not info['actor_updated'] and not report['accepted']
    equal_nested(actor, agent.actor.state_dict())
    equal_nested(target, agent.actor_target.state_dict())
    equal_nested(adam, agent.actor_optimizer.state_dict())
    assert any(not torch.equal(v, agent.critic.state_dict()[k]) for k, v in critic.items())
    assert agent.total_it == 4


def test_unrestricted_guard_matches_existing_td3_and_constrained_guard_limits_change():
    agent, buffer, states = agent_and_buffer()
    ordinary = copy.deepcopy(agent)
    rng = torch.get_rng_state()
    expected = ordinary.train(buffer, 8)
    torch.set_rng_state(rng)
    info, report = ActorStepGuard(agent, states, 100., 100.).train(buffer, 8)
    assert report['lr_scale'] == 1 and info['critic_loss'] == expected['critic_loss']
    equal_nested(agent.actor.state_dict(), ordinary.actor.state_dict())
    equal_nested(agent.actor_target.state_dict(), ordinary.actor_target.state_dict())
    equal_nested(agent.actor_optimizer.state_dict(), ordinary.actor_optimizer.state_dict())
    agent.total_it = 3
    info, report = ActorStepGuard(agent, states, mean_limit=.0002, p95_limit=.0005, backtracks=12).train(buffer, 8)
    assert report['accepted'] and report['lr_scale'] < 1
    assert report['accepted_change']['mean_mps'] <= .0002
    assert report['accepted_change']['p95_mps'] <= .0005
    assert agent.actor_optimizer.param_groups[0]['lr'] == ordinary.actor_optimizer.param_groups[0]['lr']


def test_candidate_factory_rejects_test_split():
    from scripts.ego20x10_contract_variants import contract_env
    with pytest.raises(ValueError, match='sealed test'):
        contract_env({}, CONTRACTS['baseline'], split='test')


def test_zero_calibration_weight_matches_production_frozen_td3_exactly():
    agent, buffer, states = agent_and_buffer()
    agent.actor_frozen = True
    production = copy.deepcopy(agent)
    rng = torch.get_rng_state()
    expected = production.train(buffer, 8)
    torch.set_rng_state(rng)
    info = critic_calibration_step(agent, buffer.sample(8), weight=0.)
    assert info['td_loss'] == expected['critic_loss']
    for module in ('actor', 'critic', 'actor_target', 'critic_target'):
        equal_nested(getattr(agent, module).state_dict(), getattr(production, module).state_dict())
    equal_nested(agent.critic_optimizer.state_dict(), production.critic_optimizer.state_dict())
    assert not agent.actor_optimizer.state


def test_audit_duplicate_handling_preserves_unique_trials_and_rejects_conflicts(tmp_path):
    import json
    from scripts.review_full_critic_acceptance import unique_rows
    row = {'step': 20000, 'case_id': 'one', 'candidate': 'source', 'replica': 0, 'G': 1.}
    path = tmp_path/'audit.jsonl'
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    rows, proof = unique_rows(path, 1)
    assert rows == [row] and proof['exact_duplicate_rows'] == 1
    path.write_text(json.dumps(row)+'\n'+json.dumps({**row, 'G': 2.})+'\n')
    with pytest.raises(RuntimeError, match='Conflicting'):
        unique_rows(path, 1)
