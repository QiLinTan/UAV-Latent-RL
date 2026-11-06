"""Audit saved pre/post unfreeze networks without changing production learning.

Counterfactual actions share the frozen source continuation. Consequently the
4000-step calibration is against the exact frozen policy, without the changing
continuation confound of later checkpoint comparisons.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from envs.preprocess import preprocess_state
from trainers.paired_trajectory_trace import PairedTrajectoryTrace
from scripts.audit_full_ft_v1_critic_actor import (
    ROOT as OLD_ROOT, SOURCE, actor_batch, candidates, command_metrics, dump,
    make_full_env, new_agent, q_table, replay_to_case, rollout_branch, weight_delta,
)
from scripts.reproduce_full_ft_v1_unfreeze import CKPTS, ROOT, STEPS, original_tape


def snapshot_agent(seed, step):
    prefix = CKPTS / f'seed{seed}' / f'model_{step:06d}'
    payload = torch.load(str(prefix) + '_full_snapshot.pt', map_location='cpu', weights_only=False)
    agent = new_agent(prefix)
    agent.actor_target.load_state_dict(payload['actor_target'])
    agent.critic_target.load_state_dict(payload['critic_target'])
    for name, key in (('discount', 'gamma'), ('tau', 'tau'), ('policy_noise', 'policy_noise'), ('noise_clip', 'noise_clip')):
        setattr(agent, name, float(payload['args'][key]))
    agent.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    agent.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    return agent, payload


def td_statistics(agent, sample, generator):
    s, a, ns, r, nd = (sample[key] for key in ('state', 'action', 'next_state', 'reward', 'not_done'))
    with torch.no_grad():
        noise = (torch.randn(a.shape, generator=generator) * agent.policy_noise).clamp(-agent.noise_clip, agent.noise_clip)
        next_action = agent.actor_target(ns).add(noise).clamp(-1, 1)
        next_action[:, :2] /= torch.linalg.vector_norm(next_action[:, :2], dim=1, keepdim=True).clamp(min=1.)
        tq1, tq2 = agent.critic_target(ns, next_action)
        target = r + nd * agent.discount * torch.minimum(tq1, tq2)
        q1, q2 = agent.critic(s, a)
    result = {'transitions': len(s), 'reward_scale_already_in_replay': .1,
              'mean_td_target': float(target.mean()), 'mean_twin_gap': float((q1 - q2).abs().mean())}
    for name, value in (('q1', q1), ('q2', q2)):
        error = value - target
        result[name] = {'mean': float(value.mean()), 'td_bias': float(error.mean()),
                        'td_mae': float(error.abs().mean()), 'td_rmse': float(error.square().mean().sqrt())}
    return result


def historical_rewards(seed):
    """Upper reward must be a sum of actual lower rewards, including early ends."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event = EventAccumulator(f'runs/full_ft_v1_seed{seed}', size_guidance={'scalars': 0})
    event.Reload()
    time_costs = {r.step: r.value for r in event.Scalars('env/training_episode/reward_sum/time_penalty')}
    rows = []
    for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json')):
        meta = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as d:
            ends = np.rint(d['upper_time_seconds'] * 120).astype(int)
            cumulative = np.r_[0., np.cumsum(d['lower_reward'])]
            totals = np.diff(cumulative[np.r_[0, ends]])
            error = np.max(np.abs(totals - d['upper_reward']))
            assert np.max(ends) == len(d['lower_reward'])
            assert error < 1e-5, (path, error)
            row = {'episode': meta['training_episode_index'], 'upper_steps': len(totals),
                   'lower_ticks': len(d['lower_reward']), 'max_lower_sum_error': float(error)}
            if meta['global_upper_step'] in time_costs and meta['done_reason'] != 'training_budget_end':
                expected_time = .25 * (len(d['lower_reward']) - 1) / 120
                row['time_integral_error'] = abs(time_costs[meta['global_upper_step']] - expected_time)
                assert row['time_integral_error'] < 1e-6
            rows.append(row)
    result = {'episodes': len(rows), 'upper_transitions': sum(r['upper_steps'] for r in rows),
              'lower_ticks': sum(r['lower_ticks'] for r in rows),
              'maximum_upper_vs_lower_sum_error': max(r['max_lower_sum_error'] for r in rows),
              'integration_dt_production': 1 / 120, 'upper_reward_reduction': 'sum',
              'completed_episodes_time_integral_checked': sum('time_integral_error' in r for r in rows),
              'max_episode_time_integral_error': max(r.get('time_integral_error', 0.) for r in rows),
              'episodes_detail': rows}
    dump(ROOT / f'seed{seed}' / 'historical_reward_aggregation.json', result)
    return result


def offline(seed):
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    bank = np.load(OLD_ROOT / f'seed{seed}_warmup_bank.npz')
    obs, positions = bank['observations'], bank['positions']
    source_actions = actor_batch(source, obs)
    agent4k, payload4k = snapshot_agent(seed, 4000)
    assert all(torch.equal(v, payload4k['actor'][k]) for k, v in source.actor.state_dict().items())
    assert all(torch.equal(v, payload4k['actor_target'][k]) for k, v in source.actor.state_dict().items())
    assert not payload4k['actor_optimizer']['state']
    replay = payload4k['replay']
    original = original_tape(seed)
    np.testing.assert_array_equal(replay['reward'].numpy().reshape(-1), (original['reward'][:4000] * .1).astype(np.float32))
    terminals = (replay['not_done'][:, 0] == 0) & (replay['reward'][:, 0] >= 20.)
    with torch.no_grad():
        tq1, tq2 = agent4k.critic(replay['state'][terminals], replay['action'][terminals])
    success_terminal_values = {'transitions': int(terminals.sum()), 'scaled_terminal_reward': 25.,
                              'mean_q1': float(tq1.mean()), 'mean_q2': float(tq2.mean()),
                              'q1_terminal_reward_rmse': float((tq1 - replay['reward'][terminals]).square().mean().sqrt()),
                              'q2_terminal_reward_rmse': float((tq2 - replay['reward'][terminals]).square().mean().sqrt())}
    rng = np.random.default_rng(seed + 20261012)
    probe_ids = np.sort(rng.choice(4000, 512, replace=False))
    sample = {k: v[probe_ids] for k, v in payload4k['replay'].items() if isinstance(v, torch.Tensor)}
    first, _ = snapshot_agent(seed, 4001)
    last, _ = snapshot_agent(seed, 5000)
    first_actions, last_actions = actor_batch(first, obs), actor_batch(last, obs)
    result = {'seed': seed, 'fixed_bank_size': len(obs), 'td_probe_indices': probe_ids.tolist(),
              'four_k_frozen_actor_and_target_equal_source': True, 'four_k_actor_adam_empty': True,
              'four_k_replay_rewards_exactly_original_raw_times_point_one': True,
              'four_k_success_terminal_values': success_terminal_values,
              'frames': {}}
    for step in STEPS:
        agent, payload = snapshot_agent(seed, step)
        actions = actor_batch(agent, obs)
        metrics = command_metrics(source_actions, actions, positions, np.array([3.5, 0., 1.]))
        near = bank['distances'] < 1.2
        metrics['near_goal'] = command_metrics(source_actions[near], actions[near], positions[near], np.array([3.5, 0., 1.]))
        preferences, grids = [], []
        q_records = []
        for i, state in enumerate(obs):
            grid, names = candidates(source_actions[i], last_actions[i], positions[i], np.array([3.5, 0., 1.]))
            q = q_table(agent, state, np.vstack((grid, first_actions[i])))
            best = int(np.argmax(q[:-1, 0]))
            preferences.append(q[1, 0] > q[0, 0]); grids.append(float(q[best, 0] - q[0, 0]))
            q_records.append({'state_index': i, 'source': q[0].tolist(), 'first_update': q[-1].tolist(),
                              'step5000_action': q[1].tolist(), 'best_q1': names[best],
                              'best_q1_values': q[best].tolist()})
        result['frames'][str(step)] = {
            'actor_updates': payload['actor_updates'], 'critic_updates': payload['critic_updates'],
            'actor_frozen': payload['actor_frozen'], 'actor_weight_drift': weight_delta(source.actor.state_dict(), agent.actor.state_dict()),
            'commands': metrics, 'q1_prefers_5000_action_fraction': float(np.mean(preferences)),
            'mean_best_grid_q1_advantage_over_source': float(np.mean(grids)),
            'td_probe': td_statistics(agent, sample, torch.Generator().manual_seed(seed + 726)),
            'fixed_bank_q': q_records}
        dump(ROOT / f'seed{seed}' / 'offline.json', result)
        print(f'Offline seed={seed} step={step}', flush=True)
    return result


def validations(seed):
    result_path = ROOT / f'seed{seed}' / 'validation.json'
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    done = {(r['step'], r['stage'], r['map_seed']) for r in rows}
    env, args = make_full_env()
    try:
        for step in STEPS:
            agent, _ = snapshot_agent(seed, step)
            for stage, map_seed in itertools.product((0, 3), (80, 88)):
                if (step, stage, map_seed) in done:
                    continue
                # Exact actor equality makes frozen checkpoints duplicate navigation probes.
                if step in (3999, 4000):
                    baseline = next(r for r in rows if (r['step'], r['stage'], r['map_seed']) == (0, stage, map_seed))
                    rows.append({**baseline, 'step': step, 'reuse_equal_actor_step': 0})
                else:
                    env.set_curriculum_stage_override(stage)
                    obs, _ = env.reset(seed=map_seed)
                    trace = PairedTrajectoryTrace(env, map_seed)
                    trace.install()
                    reward_total, terms, truncated = 0., False, False
                    for k in range(220):
                        state = preprocess_state(obs.reshape(-1))
                        action = agent.select_action(state)
                        obs, reward, terms, truncated, info = env.step(action)
                        trace.record_upper(action, action, reward, info, pre_observation=state)
                        reward_total += reward
                        if terms or truncated:
                            break
                    else:
                        raise AssertionError('Validation did not end under the original environment limit')
                    trace.uninstall()
                    trace.save(ROOT / f'seed{seed}' / 'validation_traces' / f'step{step}_stage{stage}',
                               done_reason=info['done_reason'], success=bool(info['success']),
                               final_goal_distance=float(info['goal_distance']))
                    rows.append({'step': step, 'stage': stage, 'map_seed': map_seed, 'outcome': info['done_reason'],
                                 'success': bool(info['success']), 'raw_return': float(reward_total),
                                 'upper_steps': k + 1, 'final_distance': float(info['goal_distance'])})
                dump(result_path, rows)
            print(f'Validation seed={seed} step={step}', flush=True)
    finally:
        env.close()
    return rows


def calibration(seed):
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    critic, _ = snapshot_agent(seed, 4000)
    first, _ = snapshot_agent(seed, 4001)
    tenth, _ = snapshot_agent(seed, 4019)
    final, _ = snapshot_agent(seed, 5000)
    frames = {step: snapshot_agent(seed, step)[0] for step in STEPS}
    path = ROOT / f'seed{seed}' / 'calibration_branches.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    finished = {(r['case_id'], r['candidate']) for r in rows}
    cases = json.loads((OLD_ROOT / f'seed{seed}_branch_cases.json').read_text())
    env, args = make_full_env()
    try:
        for case in cases:
            folder = Path(f'runs/full_ft_v1_seed{seed}/full_stage{case["stage"]}_traces/step_0000000')
            with np.load(folder / f'trace_seed{case["map_seed"]}.npz') as data:
                tape = {k: data[k].copy() for k in data.files}
            meta = json.loads((folder / f'trace_seed{case["map_seed"]}.json').read_text())
            obs = replay_to_case(env, case, tape, meta)
            state = preprocess_state(obs.reshape(-1))
            src = source.select_action(state)
            grid, names = candidates(src, final.select_action(state), env.pos[0], env.TARGET_POS)
            qgrid = q_table(critic, state, grid)
            choices = {'source': src, 'first_update': first.select_action(state),
                       'tenth_update': tenth.select_action(state), 'step5000_action': final.select_action(state),
                       'q1_best_grid': grid[int(qgrid[:, 0].argmax())],
                       'minq_best_grid': grid[int(qgrid[:, 2].argmax())]}
            for name, action in choices.items():
                if (case['case_id'], name) in finished:
                    continue
                # Critic's actor is frozen source, hence this helper continues source.
                row = rollout_branch(env, source, critic, args, case, tape, meta,
                                     'best_q1_1_then_updated', np.asarray(action))
                row.update({'candidate': name, 'critic_step': 4000,
                            'continuation_policy': 'frozen_source',
                            'continuation_actor_target_equals_source': True,
                            'kind': 'one_candidate_then_common_frozen_source'})
                row['all_frame_q1_q2_min'] = {}
                for step, frame in frames.items():
                    row['all_frame_q1_q2_min'][str(step)] = q_table(frame, state, action[None])[0].tolist()
                rows.append(row)
                dump(path, rows)
            print(f'Calibration {case["case_id"]}: six candidates, same source continuation', flush=True)
    finally:
        env.close()
    return rows


def summarize_calibration(rows):
    pairs = []
    for case_id in sorted({r['case_id'] for r in rows}):
        group = [r for r in rows if r['case_id'] == case_id]
        source = next(r for r in group if r['candidate'] == 'source')
        for candidate in group:
            if candidate['candidate'] == 'source':
                continue
            dg = candidate['discounted_scaled_return'] - source['discounted_scaled_return']
            dq = np.array(candidate['initial_q1_q2_min']) - source['initial_q1_q2_min']
            pairs.append({'case_id': case_id, 'candidate': candidate['candidate'], 'actual_return_delta': dg,
                          'q1_q2_minq_delta': dq.tolist(), 'distinguishable': abs(dg) > .05,
                          'ranking_error_q1_q2_minq': ((dq * dg) < 0).tolist(),
                          'source_outcome': source['outcome'], 'candidate_outcome': candidate['outcome']})
    summary = {'case_count': len({r['case_id'] for r in rows}), 'branches': len(rows), 'pairwise': pairs,
               'near_tie_threshold_scaled_return': .05, 'by_candidate': {}}
    for name in ('first_update', 'tenth_update', 'step5000_action', 'q1_best_grid', 'minq_best_grid'):
        subset = [p for p in pairs if p['candidate'] == name]
        identifiable = [p for p in subset if p['distinguishable']]
        summary['by_candidate'][name] = {'pairs': len(subset), 'distinguishable_pairs': len(identifiable),
            'ranking_error_counts_q1_q2_min': np.sum([p['ranking_error_q1_q2_minq'] for p in identifiable], axis=0).astype(int).tolist() if identifiable else [0, 0, 0],
            'mean_actual_return_delta': float(np.mean([p['actual_return_delta'] for p in subset])),
            'q1_prefers_candidate': sum(p['q1_q2_minq_delta'][0] > 0 for p in subset),
            'candidate_worse_actual_return': sum(p['actual_return_delta'] < -.05 for p in subset),
            'success_to_failure': sum(p['source_outcome'] == 'success' and p['candidate_outcome'] != 'success' for p in subset)}
    for name in ('source', 'q1_best_grid', 'minq_best_grid'):
        subset = [r for r in rows if r['candidate'] == name]
        q = np.array([r['initial_q1_q2_min'] for r in subset])
        g = np.array([r['discounted_scaled_return'] for r in subset])
        summary[name] = {'mean_q1_q2_min': q.mean(axis=0).tolist(), 'mean_true_full_return': float(g.mean()),
                         'mean_q_bias_q1_q2_min': (q - g[:, None]).mean(axis=0).tolist(),
                         'successes': sum(r['outcome'] == 'success' for r in subset), 'count': len(subset)}
    return summary


def noise_calibration(seed):
    """Paired common random numbers, matching TD3's frozen target smoothing.

Noise starts AFTER the forced first action. Four repeats are an uncertainty
check, not a precise Monte Carlo estimate of every action's expected return.
"""
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    critic, _ = snapshot_agent(seed, 4000)
    first, _ = snapshot_agent(seed, 4001)
    final, _ = snapshot_agent(seed, 5000)
    path = ROOT / f'seed{seed}' / 'noise_calibration.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r['case_id'], r['candidate'], r['noise_replica']) for r in rows}
    cases = [c for c in json.loads((OLD_ROOT / f'seed{seed}_branch_cases.json').read_text()) if c['map_seed'] == 80]
    env, args = make_full_env()
    try:
        for case in cases:
            folder = Path(f'runs/full_ft_v1_seed{seed}/full_stage{case["stage"]}_traces/step_0000000')
            with np.load(folder / 'trace_seed80.npz') as d:
                tape = {k: d[k].copy() for k in d.files}
            meta = json.loads((folder / 'trace_seed80.json').read_text())
            obs = replay_to_case(env, case, tape, meta)
            state = preprocess_state(obs.reshape(-1))
            src = source.select_action(state)
            grid, _ = candidates(src, final.select_action(state), env.pos[0], env.TARGET_POS)
            best = grid[int(q_table(critic, state, grid)[:, 0].argmax())]
            for replica in range(4):
                noise_seed = seed * 100000 + case['stage'] * 1000 + case['step'] * 4 + replica
                noise = np.clip(np.random.default_rng(noise_seed).normal(0, critic.policy_noise, (220, 3)),
                                -critic.noise_clip, critic.noise_clip)
                for name, action in (('source', src), ('first_update', first.select_action(state)), ('q1_best_grid', best)):
                    if (case['case_id'], name, replica) in done:
                        continue
                    obs = replay_to_case(env, case, tape, meta)
                    rewards, actions = [], []
                    for k in range(220):
                        selected = action.copy() if k == 0 else np.clip(source.select_action(preprocess_state(obs.reshape(-1))) + noise[k - 1], -1, 1)
                        selected = source.project_action(selected)
                        obs, reward, term, trunc, info = env.step(selected)
                        rewards.append(reward * args.reward_scale); actions.append(np.asarray(selected).tolist())
                        if term or trunc:
                            break
                    else:
                        raise AssertionError('Noise branch did not terminate')
                    from scripts.audit_full_ft_v1_critic_actor import discounted_return
                    rows.append({'case_id': case['case_id'], 'stage': case['stage'], 'step': case['step'],
                        'candidate': name, 'noise_replica': replica, 'noise_seed': noise_seed,
                        'policy_noise': critic.policy_noise, 'noise_clip': critic.noise_clip,
                        'initial_q1_q2_min': q_table(critic, state, action[None])[0].tolist(),
                        'discounted_scaled_return': discounted_return(rewards, args.gamma),
                        'outcome': info['done_reason'], 'actions': actions, 'rewards_scaled': rewards,
                        'continuation': 'frozen_source_plus_TD3_target_smoothing_noise'})
                    dump(path, rows)
            print(f'Noise calibration {case["case_id"]}: paired 4 replicas', flush=True)
    finally:
        env.close()
    return rows


def noise_summary(rows):
    result = {'cases': len({r['case_id'] for r in rows}), 'replicas_per_case': 4, 'branches': len(rows),
              'by_candidate': {}, 'cases_detail': []}
    for name in ('source', 'first_update', 'q1_best_grid'):
        subset = [r for r in rows if r['candidate'] == name]
        gs = np.array([r['discounted_scaled_return'] for r in subset])
        qs = np.array([r['initial_q1_q2_min'] for r in subset])
        result['by_candidate'][name] = {'mean_true_return': float(gs.mean()),
            'mean_q1_q2_min': qs.mean(axis=0).tolist(),
            'mean_q_bias_q1_q2_min': (qs - gs[:, None]).mean(axis=0).tolist(),
            'successes': sum(r['outcome'] == 'success' for r in subset), 'count': len(subset)}
    for case in sorted({r['case_id'] for r in rows}):
        group = [r for r in rows if r['case_id'] == case]
        detail = {'case_id': case, 'candidates': {}}
        for name in ('source', 'first_update', 'q1_best_grid'):
            subset = [r for r in group if r['candidate'] == name]
            g = [r['discounted_scaled_return'] for r in subset]
            detail['candidates'][name] = {'mean_return': float(np.mean(g)), 'std_return': float(np.std(g, ddof=1)),
                                        'returns': g, 'q1_q2_min': subset[0]['initial_q1_q2_min']}
        result['cases_detail'].append(detail)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(43, 45), required=True)
    parser.add_argument('--phase', choices=('offline', 'validation', 'calibration', 'noise', 'summary', 'historical'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = ROOT / f'seed{args.seed}'
    if args.phase == 'offline':
        offline(args.seed)
    elif args.phase == 'historical':
        historical_rewards(args.seed)
    elif args.phase == 'validation':
        validations(args.seed)
    elif args.phase == 'calibration':
        calibration(args.seed)
    elif args.phase == 'noise':
        noise_calibration(args.seed)
    else:
        result = {'seed': args.seed}
        for key, file in (('reproduction', 'reproduction_summary.json'), ('offline', 'offline.json'),
                          ('validation', 'validation.json'), ('historical_reward', 'historical_reward_aggregation.json')):
            result[key] = json.loads((root / file).read_text())
        result['calibration'] = summarize_calibration(json.loads((root / 'calibration_branches.json').read_text()))
        result['target_noise_check'] = noise_summary(json.loads((root / 'noise_calibration.json').read_text()))
        dump(root / 'summary.json', result)
        print(json.dumps(result['calibration'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
