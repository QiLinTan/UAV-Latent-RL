"""Two-level fixed-policy Full critic acceptance and projected action gradients.

All candidates change only the first upper action; the fixed source actor then
continues until the original episode ends. G discounts once per 15Hz transition
and scales the original sum of 120Hz rewards exactly once. No actor is updated.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from envs.preprocess import preprocess_state
from scripts.run_full_ft_v2 import dump, differences, sha
from scripts.run_full_critic_frozen import DEFAULT_ROOT, REFERENCE, production_hashes
from scripts.audit_full_ft_v1_critic_actor import make_full_env, replay_to_case, candidates, q_table, discounted_return

AUDIT_STEPS = (4000, 10000, 20000)
GRAD_STEPS = (.05, .15, .30)
TIE = .05
LOCAL_TIE = .005
NOISE_REPLICAS = 4
ACCEPTANCE = {'value_mae_max': 2., 'value_abs_bias_max': 1.,
              'ranking_agreement_min': .8, 'ranking_pairs_min': 200,
              'ranking_states_min': 30, 'ranking_states_per_stage_min': 10,
              'gradient_primary_step': .15, 'gradient_agreement_min': .7,
              'gradient_pairs_min': 30, 'gradient_mean_gain_min': 0.,
              'gradient_states_per_stage_min': 5, 'local_headroom_states_min': 10,
              'gradient_recovered_headroom_fraction_min': .5,
              'gradient_q_ascent_fraction_min': .8,
              'map_bootstrap_ci_lower_min': .5, 'return_near_tie_threshold': TIE,
              'local_gradient_return_near_tie_threshold': LOCAL_TIE}


def load_agent(root, step):
    cfg = json.loads((root / 'configuration.json').read_text())
    verification = json.loads((root / 'prefix_verification.json').read_text())
    if not verification['all_fields_exactly_equal'] or verification['reference_sha256'] != sha(REFERENCE):
        raise RuntimeError('Reference 4k state was not verified or has since changed')
    p = torch.load(Path(cfg['checkpoint_directory']) / f'model_{step:06d}_full_snapshot.pt', map_location='cpu', weights_only=False)
    original = torch.load(REFERENCE, map_location='cpu', weights_only=False)
    if p['actor_updates'] != 0 or not p['actor_frozen']:
        raise RuntimeError('Audit requires a fully frozen source policy')
    for name in ('actor', 'actor_target'):
        if differences(p[name], original['actor']):
            raise RuntimeError(f'{name} changed from the source')
    if p['critic_optimizer']['param_groups'][0]['lr'] != 3e-4:
        raise RuntimeError('Critic LR is not the original 3e-4')
    agent = TD3UpperRecurrent(917, 3, 1., base_state_dim=117, history_length=8,
                             history_frame_dim=100, hidden_dim=64,
                             project_planar_action=True, device=torch.device('cpu'))
    for name in ('actor', 'actor_target', 'critic', 'critic_target'):
        getattr(agent, name).load_state_dict(p[name])
        getattr(agent, name).requires_grad_(False)
    agent.actor_frozen = True
    for name, key in [('discount', 'gamma'), ('tau', 'tau'), ('policy_noise', 'policy_noise'), ('noise_clip', 'noise_clip')]:
        setattr(agent, name, p['args'][key])
    return agent, p


def build_cases():
    cases = []
    for stage, seed in itertools.product((0, 3), range(80, 90)):
        stem = Path(f'runs/full_ft_v1_seed43/full_stage{stage}_traces/step_0000000/trace_seed{seed}')
        meta = json.loads(stem.with_suffix('.json').read_text())
        with np.load(stem.with_suffix('.npz')) as d:
            n = len(d['upper_action'])
            positions = np.vstack([meta['initial_position'], d['upper_position'][:-1]])
            distance = np.linalg.norm(positions - meta['target_position'], axis=1)
        points = [(0, 'initial'), (n - 1, 'preterminal')]
        if n > 60:
            points.append((60, 'mid60'))
        near = np.flatnonzero(distance < 1.2)
        if len(near):
            points.append((int(near[0]), 'near_goal'))
        groups = defaultdict(list)
        for index, phase in points:
            groups[index].append(phase)
        for index in sorted(groups):
            cases.append({'case_id': f'stage{stage}_seed{seed}_upper{index}', 'training_seed': 43,
                          'stage': stage, 'map_seed': seed, 'step': index, 'phases': groups[index],
                          'distance': float(distance[index]), 'baseline_outcome': meta['done_reason'],
                          'tape_stem': str(stem), 'tape_sha256': sha(stem.with_suffix('.npz')),
                          'metadata_sha256': sha(stem.with_suffix('.json'))})
    return cases


def project(action):
    a = np.clip(np.asarray(action, dtype=np.float32), -1., 1.).copy()
    norm = np.linalg.norm(a[:2])
    if norm > 1.:
        a[:2] /= norm
    return a


def action_gradient(agent, state, action):
    candidate = torch.tensor(action[None], dtype=torch.float32, requires_grad=True)
    s = torch.tensor(state[None], dtype=torch.float32)
    q = agent.critic.Q1(s, candidate)
    gradient = torch.autograd.grad(q.sum(), candidate)[0].detach().numpy()[0]
    if not np.isfinite(gradient).all():
        raise RuntimeError('Nonfinite Q1 action gradient')
    return gradient


def proposed_actions(agent, state, position, target):
    source = project(agent.select_action(state))
    grad = action_gradient(agent, state, source)
    direction = grad / max(float(np.linalg.norm(grad)), 1e-12)
    selected = {'source': source, 'stop': np.zeros(3, dtype=np.float32)}
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.eye(3)[axis] * .3 * sign
            selected[f'axis{axis}_{sign:+d}'] = project(source + delta)
    for size in GRAD_STEPS:
        for sign, name in ((1, 'plus'), (-1, 'minus')):
            selected[f'gradient_{name}_{size:.2f}'] = project(source + sign * size * direction)
    grid, _ = candidates(source, source, position, target)
    q = q_table(agent, state, grid)
    selected['q1_best_grid'] = project(grid[q[:, 0].argmax()])
    selected['minq_best_grid'] = project(grid[q[:, 2].argmax()])
    return selected, grad


def rollout(env, agent, case, tape, meta, action, noise=None):
    obs = replay_to_case(env, case, tape, meta)
    rewards, outcomes = [], []
    for k in range(220):
        state = preprocess_state(obs.reshape(-1))
        chosen = action if k == 0 else agent.select_action(state)
        if k > 0 and noise is not None:
            chosen = chosen + noise[k - 1]
        chosen = project(chosen)
        obs, reward, terminated, truncated, info = env.step(chosen)
        rewards.append(float(reward) * .1)
        outcomes.append({'action': chosen.tolist(), 'scaled_reward': rewards[-1],
                         'position': env.pos[0].tolist(), 'velocity': env.vel[0].tolist(),
                         'command_velocity': env.command_velocity.tolist()})
        if terminated or truncated:
            break
    else:
        raise RuntimeError('Branch did not terminate under original 12s environment limit')
    return {'G': discounted_return(rewards, agent.discount), 'outcome': info['done_reason'],
            'success': bool(info['success']), 'upper_steps': len(rewards),
            'final_distance': float(info['goal_distance']), 'trace': outcomes}


def cached_rollout(root, env, agent, case, tape, meta, action, noise, protocol, replica):
    # Same state/action/protocol has the same G across critic checkpoints.
    key = hashlib.sha256(project(action).tobytes()).hexdigest()[:20]
    cache = root / 'returns' / case['case_id'] / f'{protocol}_rep{replica}_{key}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    result = rollout(env, agent, case, tape, meta, action, noise)
    dump(cache, result)
    return result


def evaluate(root, steps, phase, case_limit=None):
    torch.set_num_threads(1)
    output = root / ('audit_smoke' if case_limit is not None else 'audit')
    output.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    if case_limit is not None:
        cases = cases[:case_limit]
    cfg = json.loads((root / 'configuration.json').read_text())
    production = production_hashes()
    if cfg.get('production_inputs') is not None and cfg['production_inputs'] != production:
        raise RuntimeError('Production code/map inputs differ from the training run; audit stopped')
    manifest = {'case_plan': cases, 'source_actor_sha256': cfg['source_sha256'],
                'reference_sha256': sha(REFERENCE), 'production_code_sha256': production,
                'protocol': {'first_action': 'candidate once', 'continuation': 'unchanged source actor',
                             'reward_scale': .1, 'gamma': .99, 'lower_reward_dt': 1/120,
                             'upper_discount_dt': 1/15, 'episode_limit': 'original remaining 12s budget',
                             'noise_sigma': .2, 'noise_clip': .5, 'noise_replicas': NOISE_REPLICAS},
                'acceptance': ACCEPTANCE, 'smoke_only': case_limit is not None}
    checkpoints = {str(step): sha(Path(cfg['checkpoint_directory']) / f'model_{step:06d}_full_snapshot.pt') for step in steps}
    path = output / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        old_checkpoints = old.pop('checkpoint_sha256', {})
        if differences(old, json.loads(json.dumps(manifest))):
            raise RuntimeError('Audit inputs/code changed; refusing to mix cached evidence')
        if any(k in checkpoints and checkpoints[k] != v for k,v in old_checkpoints.items()):
            raise RuntimeError('A previously audited critic checkpoint changed')
        checkpoints = {**old_checkpoints, **checkpoints}
    manifest['checkpoint_sha256'] = checkpoints
    dump(path, manifest)
    env, _ = make_full_env()
    try:
        for step in steps:
            agent, payload = load_agent(root, step)
            assert payload['args']['gamma'] == .99 and payload['args']['reward_scale'] == .1
            assert agent.policy_noise == .2 and agent.noise_clip == .5
            frame = output / f'step_{step:06d}_{phase}.jsonl'
            existing = [json.loads(line) for line in frame.read_text().splitlines()] if frame.exists() else []
            completed = {(r['case_id'], r['candidate'], r['replica']) for r in existing}
            for case in cases:
                stem = Path(case['tape_stem'])
                with np.load(stem.with_suffix('.npz')) as d:
                    tape = {k: d[k].copy() for k in d.files}
                meta = json.loads(stem.with_suffix('.json').read_text())
                obs = replay_to_case(env, case, tape, meta)
                state = preprocess_state(obs.reshape(-1))
                selected, gradient = proposed_actions(agent, state, env.pos[0], env.TARGET_POS)
                if phase == 'noise':
                    selected = {name: selected[name] for name in ('source', 'q1_best_grid', 'gradient_plus_0.15')}
                replicas = range(NOISE_REPLICAS) if phase == 'noise' else (0,)
                for name, action in selected.items():
                    q = q_table(agent, state, action[None])[0]
                    for replica in replicas:
                        if (case['case_id'], name, replica) in completed:
                            continue
                        seed = 20261013 + case['map_seed'] * 10000 + case['stage'] * 1000 + case['step'] * 4 + replica
                        noise = (np.clip(np.random.default_rng(seed).normal(0, .2, (220, 3)), -.5, .5)
                                 if phase == 'noise' else None)
                        result = cached_rollout(output, env, agent, case, tape, meta, action, noise, phase, replica)
                        row = {'step': step, 'case_id': case['case_id'], 'stage': case['stage'],
                               'map_seed': case['map_seed'], 'phases': case['phases'], 'candidate': name,
                               'action': action.tolist(), 'Q': q.tolist(), 'G': result['G'],
                               'outcome': result['outcome'], 'success': result['success'],
                               'q1_action_gradient': gradient.tolist(), 'replica': replica,
                               'noise_seed': seed if noise is not None else None, 'protocol': phase}
                        with frame.open('a') as f:
                            f.write(json.dumps(row, allow_nan=False) + '\n')
                print(f'[Frozen critic audit] {phase} step={step} {case["case_id"]}', flush=True)
    finally:
        env.close()


def calibration(rows):
    q = np.asarray([r['Q'] for r in rows]); g = np.asarray([r['G'] for r in rows])
    return {name: {'n': len(rows), 'mean_Q': float(q[:, i].mean()), 'mean_G': float(g.mean()),
                   'bias': float((q[:, i] - g).mean()), 'mae': float(np.abs(q[:, i] - g).mean()),
                   'rmse': float(np.sqrt(np.mean((q[:, i] - g)**2)))}
            for i, name in enumerate(('q1', 'q2', 'minq'))}


def map_bootstrap(pairs, index):
    if not pairs:
        return None
    groups = defaultdict(list)
    for r in pairs:
        groups[r['map_seed']].append(r)
    totals = {k: np.array([sum(r['correct'][index] for r in rows), len(rows)], dtype=float)
              for k, rows in groups.items()}
    keys = sorted(totals)
    rng = np.random.default_rng(20261013)
    estimates = []
    for _ in range(1000):
        counts = sum((totals[keys[i]] for i in rng.integers(len(keys), size=len(keys))), np.zeros(2))
        estimates.append(counts[0] / counts[1])
    return np.quantile(estimates, [.025, .975]).tolist()


def ranking(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['case_id']].append(r)
    pairs, state_accuracy = [], []
    for state, group in groups.items():
        local = []
        # Deduplicate actions; aliases must not multiply evidence.
        unique = {project(r['action']).tobytes(): r for r in group}
        for a, b in itertools.combinations(unique.values(), 2):
            dg = a['G'] - b['G']
            if abs(dg) <= TIE:
                continue
            dq = np.asarray(a['Q']) - b['Q']
            p = {'case_id': state, 'stage': a['stage'], 'map_seed': a['map_seed'],
                 'correct': (dq * dg > 0).tolist(), 'q_tie': (np.abs(dq) <= 1e-6).tolist()}
            # Q near-ties are no guidance, not correct signs.
            p['correct'] = [c and not t for c, t in zip(p['correct'], p['q_tie'])]
            pairs.append(p); local.append(p)
        if local:
            state_accuracy.append({'case_id': state, 'stage': group[0]['stage'],
                                   'accuracy': np.mean([r['correct'] for r in local], axis=0).tolist()})
    by_stage = {stage: sum(r['stage'] == stage for r in state_accuracy) for stage in (0, 3)}
    sufficient = (len(pairs) >= ACCEPTANCE['ranking_pairs_min']
                  and len(state_accuracy) >= ACCEPTANCE['ranking_states_min']
                  and min(by_stage.values()) >= ACCEPTANCE['ranking_states_per_stage_min'])
    result = {'distinguishable_pairs': len(pairs), 'states_with_distinguishable_pairs': len(state_accuracy),
              'states_per_stage': by_stage, 'sufficient_evidence': sufficient}
    for i, name in enumerate(('q1', 'q2', 'minq')):
        ci = map_bootstrap(pairs, i)
        agreement = float(np.mean([r['correct'][i] for r in pairs])) if pairs else None
        result[name] = {'agreement': agreement, 'map_cluster_bootstrap_95ci': ci,
                        'equal_state_mean_agreement': float(np.mean([r['accuracy'][i] for r in state_accuracy])) if state_accuracy else None,
                        'q_near_tie_count': sum(r['q_tie'][i] for r in pairs)}
    return result


def gradient_quality(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row['case_id']][row['candidate']] = row
    result = {}
    for size in GRAD_STEPS:
        pairs, gains, nonzero, headroom, recovered = [], [], 0, 0, 0
        for case, group in groups.items():
            plus, minus = group[f'gradient_plus_{size:.2f}'], group[f'gradient_minus_{size:.2f}']
            source = group['source']
            local_better = any(r['G'] > source['G'] + LOCAL_TIE
                               and np.linalg.norm(np.asarray(r['action']) - source['action']) <= .35
                               for r in group.values())
            headroom += int(local_better)
            recovered += int(local_better and plus['G'] > source['G'] + LOCAL_TIE)
            if np.linalg.norm(np.asarray(plus['action']) - minus['action']) < 1e-8:
                continue
            nonzero += 1
            gains.append(plus['G'] - source['G'])
            dg = plus['G'] - minus['G']
            if abs(dg) <= LOCAL_TIE:
                continue
            pairs.append({'case_id': case, 'map_seed': plus['map_seed'], 'stage': plus['stage'],
                          'correct': [dg > 0], 'q_ascent': plus['Q'][0] > minus['Q'][0] + 1e-6})
        ci = map_bootstrap(pairs, 0)
        agreement = float(np.mean([p['correct'][0] for p in pairs])) if pairs else None
        result[f'{size:.2f}'] = {'effective_projected_directions': nonzero,
                               'distinguishable_gradient_pairs': len(pairs), 'agreement': agreement,
                               'distinguishable_states_per_stage': {s: sum(p['stage']==s for p in pairs) for s in (0,3)},
                               'states_with_measurable_local_headroom': headroom,
                               'gradient_recovered_headroom_fraction': recovered/headroom if headroom else None,
                               'map_cluster_bootstrap_95ci': ci,
                               'mean_G_plus_minus_source': float(np.mean(gains)) if gains else None,
                               'Q1_plus_above_minus_fraction': float(np.mean([p['q_ascent'] for p in pairs])) if pairs else None}
    return result


def source_value_pass(stats):
    return all(stats[name]['mae'] <= ACCEPTANCE['value_mae_max']
               and abs(stats[name]['bias']) <= ACCEPTANCE['value_abs_bias_max'] for name in ('q1', 'q2'))


def verdict(source_stats, all_stats, ranks, gradients, complete):
    source_pass, candidate_pass = source_value_pass(source_stats), source_value_pass(all_stats)
    qr = ranks['q1']; gr = gradients['0.15']
    rank_ready = ranks['sufficient_evidence']
    grad_ready = (gr['distinguishable_gradient_pairs'] >= ACCEPTANCE['gradient_pairs_min']
                  and min(gr['distinguishable_states_per_stage'].values()) >= ACCEPTANCE['gradient_states_per_stage_min']
                  and gr['states_with_measurable_local_headroom'] >= ACCEPTANCE['local_headroom_states_min'])
    ranking_pass = (rank_ready and qr['agreement'] >= .8 and qr['map_cluster_bootstrap_95ci'][0] > .5)
    gradient_pass = (grad_ready and gr['agreement'] >= .7 and gr['map_cluster_bootstrap_95ci'][0] > .5
                     and gr['mean_G_plus_minus_source'] >= 0.
                     and gr['Q1_plus_above_minus_fraction'] >= ACCEPTANCE['gradient_q_ascent_fraction_min']
                     and gr['gradient_recovered_headroom_fraction'] >= ACCEPTANCE['gradient_recovered_headroom_fraction_min'])
    return {'source_value_calibrated': source_pass, 'candidate_value_calibrated': candidate_pass,
            'level1_value_pass': complete and source_pass and candidate_pass,
            'ranking_evidence_sufficient': rank_ready, 'gradient_evidence_sufficient': grad_ready,
            'level2_guidance_pass': complete and ranking_pass and gradient_pass,
            'level2_status': 'insufficient_evidence' if not complete or not rank_ready or not grad_ready
                             else 'pass' if ranking_pass and gradient_pass else 'fail',
            'actor_utilization_scope': 'local action-space guidance only; actor was never updated'}


def noise_control_summary(rows, expected_cases, step):
    """Count each Monte Carlo replica once; conflicting repeats are invalid evidence."""
    unique = {}
    for row in rows:
        if row['step'] != step:
            raise RuntimeError(f'Noise record belongs to a different checkpoint: {row["step"]}')
        key = (row['case_id'], row['candidate'], row['replica'])
        if key in unique and unique[key] != row:
            raise RuntimeError(f'Conflicting duplicate noise record: {key}')
        unique.setdefault(key, row)
    expected = {(case, candidate, replica) for case in expected_cases
                for candidate in ('source', 'q1_best_grid', 'gradient_plus_0.15')
                for replica in range(NOISE_REPLICAS)}
    groups = defaultdict(list)
    for row in unique.values():
        groups[(row['case_id'], row['candidate'])].append(row)
    averaged = [{**group[0], 'G': float(np.mean([r['G'] for r in group])),
                 'G_std': float(np.std([r['G'] for r in group], ddof=1)) if len(group)>1 else None,
                 'replicas': len(group)} for group in groups.values()]
    source = [r for r in averaged if r['candidate'] == 'source']
    return {'complete': set(unique) == expected,
            'input_records': len(rows), 'unique_records': len(unique),
            'expected_records': len(expected), 'duplicate_records_ignored': len(rows)-len(unique),
            'missing_records': len(expected-set(unique)), 'unexpected_records': len(set(unique)-expected),
            'source_calibration': calibration(source) if source else None,
            'per_candidate': averaged,
            'role': 'four common-noise replicas check smoothing confound; not a high-precision MC estimate'}


def summarize(root, steps=AUDIT_STEPS, smoke=False):
    output = root / ('audit_smoke' if smoke else 'audit')
    manifest = json.loads((output / 'manifest.json').read_text())
    expected = {r['case_id'] for r in manifest['case_plan']}
    report = ['# Full冻结Actor：Critic两层验收', '',
              '训练累计20k包含原4k，新增16k；Critic LR保持3e-4。Actor及其Target全程不更新。G只改变第一步动作，此后固定源Actor，使用原Full剩余时限、逐上层转移gamma=0.99和一次奖励缩放0.1。', '',
              '以下为固定验证状态/动作库上的诊断，不能推断整个Full状态空间。噪声控制与确定性主协议分别报告。动作对在同一状态/地图内相关，CI按地图seed聚类bootstrap。', '',
              '| 步数 | 源动作Q1 MAE / bias | 全候选Q1 MAE | 可区分动作对 | Q1排序一致率 | 梯度可区分对 | 梯度一致率 | 第一层 | 第二层 |',
              '| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |']
    summary = {'acceptance': ACCEPTANCE, 'steps': {}, 'smoke_only': smoke}
    for step in steps:
        path = output / f'step_{step:06d}_deterministic.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        actual = {r['case_id'] for r in rows}
        required_names = {'source', 'stop', 'q1_best_grid', 'minq_best_grid'}
        required_names.update(f'axis{axis}_{sign:+d}' for axis in range(3) for sign in (-1, 1))
        required_names.update(f'gradient_{sign}_{size:.2f}' for size in GRAD_STEPS for sign in ('plus', 'minus'))
        complete = (not smoke and actual == expected
                    and all({r['candidate'] for r in rows if r['case_id'] == case} == required_names for case in expected))
        source = [r for r in rows if r['candidate'] == 'source']
        all_unique = list({(r['case_id'], project(r['action']).tobytes()): r for r in rows}.values())
        src_stats, all_stats = calibration(source), calibration(all_unique)
        ranks, gradients = ranking(rows), gradient_quality(rows)
        decisions = verdict(src_stats, all_stats, ranks, gradients, complete)
        result = {'source_calibration': src_stats, 'all_candidate_calibration': all_stats,
                  'ranking': ranks, 'gradient': gradients, 'verdict': decisions,
                  'per_stage_source_calibration': {s: calibration([r for r in source if r['stage'] == s]) for s in (0, 3) if any(r['stage']==s for r in source)},
                  'per_phase_source_calibration': {p: calibration([r for r in source if p in r['phases']])
                                                 for p in ('initial','mid60','near_goal','preterminal') if any(p in r['phases'] for r in source)}}
        noise_path = output / f'step_{step:06d}_noise.jsonl'
        if noise_path.exists():
            noise_rows = [json.loads(line) for line in noise_path.read_text().splitlines()]
            result['target_noise_control'] = noise_control_summary(noise_rows, expected, step)
        summary['steps'][str(step)] = result
        rate = lambda x: 'NA' if x is None else f'{x:.1%}'
        grad = gradients['0.15']
        report.append(f'| {step} | {src_stats["q1"]["mae"]:.3f} / {src_stats["q1"]["bias"]:.3f} | {all_stats["q1"]["mae"]:.3f} | {ranks["distinguishable_pairs"]} | {rate(ranks["q1"]["agreement"])} | {grad["distinguishable_gradient_pairs"]} | {rate(grad["agreement"])} | {decisions["level1_value_pass"]} | {decisions["level2_status"]} |')
    report += ['', '## 预设目标噪声对照', '',
               '每个状态三种首步候选（源动作、Q1网格最高动作、正梯度0.15动作），后续冻结源Actor加原TD3目标噪声sigma=0.2、clip=0.5，四个共同随机副本。第一步不加噪声。源动作校准对比使用各状态四个不同副本的平均G。', '',
               '| 步数 | 唯一记录 / 预期 | 忽略的相同重复记录 | 完整 | 确定性源Q1 MAE | 噪声源Q1 MAE / bias | 噪声源Q2 MAE / bias |',
               '| --- | --- | ---: | --- | ---: | --- | --- |']
    noise_complete = True
    for step in steps:
        result = summary['steps'][str(step)]
        noise = result.get('target_noise_control')
        noise_complete = noise_complete and noise is not None and noise['complete']
        stats = noise.get('source_calibration') if noise else None
        if stats:
            q1, q2 = stats['q1'], stats['q2']
            report.append(f'| {step} | {noise["unique_records"]} / {noise["expected_records"]} | {noise["duplicate_records_ignored"]} | {noise["complete"]} | {result["source_calibration"]["q1"]["mae"]:.3f} | {q1["mae"]:.3f} / {q1["bias"]:.3f} | {q2["mae"]:.3f} / {q2["bias"]:.3f} |')
        else:
            report.append(f'| {step} | NA | NA | False | {result["source_calibration"]["q1"]["mae"]:.3f} | NA | NA |')
    summary['preset_noise_control_complete'] = bool(noise_complete)
    report += ['', f'预设噪声对照全部完成：{bool(noise_complete)}。相同重复日志按状态/候选/副本去重；内容冲突立即拒绝汇总。原始JSONL和回报缓存保持不变，重复记录不增加独立证据。', '',
               '四副本可检查目标平滑与确定性回报定义的差别；不是高精度期望估计。噪声候选只有三种，未包含负梯度及全动作库，因此不据此重新判定第二层排序/梯度验收。']
    report += ['', 'Q1是原TD3 Actor目标，第二层主要按Q1验收；Q2/minQ作为并列诊断。梯度检查为源动作沿正/负Q1梯度投影扰动，未训练或更新任何Actor。第一层通过不保证第二层通过；第二层通过也不证明实际Actor参数更新稳定。', '',
               '若可区分动作对或梯度对不足，第二层标为insufficient_evidence，不计作通过。全候选排序近似平局阈值为缩放G差0.05；局部梯度使用较小的0.005阈值，检查微小动作扰动的确定性回报变化；Q近似平局阈值1e-6。重复动作去重，不把别名计作额外证据。', '',
               '固定地图seed80–89，共十个seed、Stage0/3二十个任务，初始/中途/近目标/终止前分层取状态，保留源策略失败任务。没有使用环境40–79作新增验证。', '',
               '该主实验同时增加经验和Critic更新次数。若4k→20k校准/排序改善，可说明同机制延长训练有效，不能单独归因为数据量增加；分离纯经验不足还需要固定4k Replay的额外更新对照。', '',
               '详细Q1/Q2/minQ、分Stage/阶段校准、地图bootstrap区间、三个梯度步长、全部回报和噪声控制见summary.json及逐分支JSON。原始20k Critic没有按导航成功率恢复到早期模型。']
    dump(output / 'summary.json', summary)
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    print(f'[Frozen critic acceptance report] {output / "report.md"}', flush=True)
    return summary


def audit_all(root):
    for phase in ('deterministic', 'noise'):
        evaluate(root, AUDIT_STEPS, phase)
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--phase', choices=('deterministic', 'noise', 'summary', 'all'), default='all')
    parser.add_argument('--steps', type=int, nargs='+', default=list(AUDIT_STEPS))
    parser.add_argument('--case_limit', type=int, help='Smoke only: outputs audit_smoke and cannot pass acceptance')
    cli = parser.parse_args()
    if cli.case_limit is not None and cli.case_limit < 1:
        parser.error('case_limit must be positive')
    if cli.phase == 'all':
        for phase in ('deterministic', 'noise'):
            evaluate(cli.output_root, cli.steps, phase, cli.case_limit)
        summarize(cli.output_root, cli.steps, cli.case_limit is not None)
    elif cli.phase == 'summary':
        summarize(cli.output_root, cli.steps, cli.case_limit is not None)
    else:
        evaluate(cli.output_root, cli.steps, cli.phase, cli.case_limit)


if __name__ == '__main__':
    main()
