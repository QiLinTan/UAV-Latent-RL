"""Verify saved Full P0 transitions and exercise updates on an isolated copy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer
from data.stratified_replay_buffer import DifficultyStratifiedReplayBuffer
from trainers.full_scratch import restore_buffer, fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='full_scratch_curr_v1_seed43')
    args = parser.parse_args()
    checkpoint = Path('checkpoints') / args.run
    root = Path('runs') / args.run / 'p0_audit'
    snapshot_path = checkpoint / 'training_state.pt'
    before = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    saved = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    initial = json.loads((checkpoint / 'initialization.json').read_text())
    agent = TD3UpperRecurrent.__new__(TD3UpperRecurrent)
    agent.__dict__.update(saved['agent'])
    assert initial['pretrained_source'] is None
    assert agent.total_it == 0 and not agent.actor_optimizer.state and not agent.critic_optimizer.state
    assert fingerprint(agent.actor) == initial['actor']
    assert fingerprint(agent.critic) == initial['critic']
    metadata = [json.loads(p.read_text()) for p in sorted((root / 'traces').glob('*.json'))]
    completed = [m for m in metadata if m['done_reason'] != 'p0_budget_boundary']
    resets = [json.loads(line) for line in (root / 'resets.jsonl').read_text().splitlines()]
    lower_ticks = Counter()
    all_clearances, all_tracking, all_actions, all_policy, finals = [], [], [], [], []
    guidance = 0
    successes = 0
    for meta in metadata:
        stem = meta['trace_id']
        trace = np.load(root / 'traces' / f'{stem}.npz')
        replay = np.load(root / 'traces' / f'{stem}_replay.npz')
        n = meta['upper_steps']
        assert trace['upper_pre_observation'].shape == replay['next_state'].shape == (n, 917)
        np.testing.assert_allclose(trace['upper_action'], replay['replay_action'])
        np.testing.assert_allclose(trace['upper_reward'] * initial['args']['reward_scale'], replay['replay_reward'], rtol=1e-6)
        np.testing.assert_allclose(trace['upper_pre_observation'][1:], replay['next_state'][:-1])
        np.testing.assert_allclose(trace['upper_pre_observation'][:, 117:].reshape(n, 8, 100)[:, 1:],
                                   replay['next_state'][:, 117:].reshape(n, 8, 100)[:, :-1])
        offset = 0
        for i, ticks in enumerate(replay['lower_ticks']):
            ticks = int(ticks)
            lower_ticks[ticks] += 1
            assert ticks == 8 or replay['terminated'][i] or replay['truncated'][i]
            np.testing.assert_allclose(trace['lower_reward'][offset:offset+ticks].sum(), trace['upper_reward'][i], rtol=1e-9)
            assert np.all(trace['lower_packet_version'][offset:offset+ticks] == trace['upper_packet_version'][i])
            offset += ticks
        assert offset == meta['lower_steps'] and np.all(trace['lower_packet_valid'])
        np.testing.assert_allclose(trace['lower_executed_rpm'], trace['lower_commanded_rpm_clipped'], rtol=1e-6)
        all_clearances.extend(trace['lower_clearance'])
        all_tracking.extend(np.linalg.norm(trace['lower_position'] - trace['lower_reference_position'], axis=1))
        all_actions.extend(trace['upper_action'])
        all_policy.extend(trace['upper_policy_action'])
        if meta in completed:
            guidance += min(np.linalg.norm(np.array(meta['initial_position']) - meta['target_position']), trace['lower_goal_distance'].min()) < 1.2
            finals.append(meta['final_goal_distance'])
        if meta['success']:
            successes += n
    replay = ReplayBuffer(917, 3, max_size=saved['replay']['max_size'])
    restore_buffer(replay, saved['replay'])
    success = DifficultyStratifiedReplayBuffer(917, 3, max_size=initial['args']['success_replay_size'])
    restore_buffer(success, saved['success'])
    assert success.size == successes
    torch.set_num_threads(2)
    np.random.seed(31415)
    torch.manual_seed(31415)
    pre_actor, pre_critic = fingerprint(agent.actor), fingerprint(agent.critic)
    updates = [agent.train(replay, batch_size=256, success_replay_buffer=success,
                           success_replay_fraction=initial['args']['success_replay_fraction']) for _ in range(8)]
    assert all(all(v is None or np.isfinite(v) for v in row.values()) for row in updates)
    assert sum(row['actor_updated'] for row in updates) == 4
    assert fingerprint(agent.actor) != pre_actor and fingerprint(agent.critic) != pre_critic
    assert agent.actor_optimizer.state and agent.critic_optimizer.state
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == before
    (root / 'isolated_update_probe.json').write_text(json.dumps(updates, indent=2) + '\n')
    act = np.asarray(all_actions)
    pol = np.asarray(all_policy)
    summary = dict(upper_steps=saved['steps'], lower_steps=sum(k*v for k,v in lower_ticks.items()),
                   lower_ticks_histogram=dict(lower_ticks), replay_size=replay.size,
                   success_replay_size=success.size, completed_episodes=len(completed),
                   outcomes=dict(Counter(m['done_reason'] for m in completed)),
                   partial_episodes=len(metadata)-len(completed),
                   reset_sources=dict(Counter(r['source'] for r in resets)),
                   start_level=saved['start']['level'], obstacle_stage=saved['obstacle']['stage'],
                   guidance_entry_fraction=guidance/len(completed),
                   final_distance_mean=float(np.mean(finals)),
                   lower_tracking_rmse=float(np.sqrt(np.mean(np.square(all_tracking)))),
                   clearance_quantiles=dict(zip(('min','p05','p50','p95'), np.quantile(all_clearances,[0,.05,.5,.95]).tolist())),
                   executed_planar_boundary_fraction=float(np.mean(np.linalg.norm(act[:,:2],axis=1)>=.999)),
                   random_action_vertical_mean=float(np.mean(act[:,2])),
                   untrained_actor_mean_action=pol.mean(axis=0).tolist(),
                   production_network_updates=0, isolated_critic_updates=8, isolated_actor_updates=4,
                   initial_actor=initial['actor'], initial_critic=initial['critic'],
                   snapshot_sha256=before, assertions_passed=True)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
