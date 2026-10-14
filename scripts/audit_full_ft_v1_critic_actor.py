"""Read-only Full FT actor drift, critic ranking and matched-state return audit.

Uses saved weights and Full trajectories; never updates an optimizer. Direct Q
calibration forces exactly one candidate action, then one common online policy.
Longer forced prefixes use prefix reward plus a bootstrap, not initial-action Q.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from envs.preprocess import preprocess_state
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv

BASE = 'upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1'
SOURCE = Path('checkpoints/startobscurr_v1_selected_20261010')
ROOT = Path('analysis_outputs/full_ft_v1_critic_actor_audit_20261012')
SCALE = np.array([.8, .8, .3])
CHECKPOINT_STEPS = (5000, 10000, 20000, 40000)
DIAGNOSTIC_MAP_SEEDS = (80, 88)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_agent(prefix):
    agent = TD3UpperRecurrent(917, 3, 1., base_state_dim=117,
        history_length=8, history_frame_dim=100, hidden_dim=64,
        project_planar_action=True, device=torch.device('cpu'))
    agent.load_actor_only(str(prefix))
    critic_path = Path(str(prefix) + '_critic')
    if critic_path.exists():
        agent.critic.load_state_dict(torch.load(critic_path, map_location='cpu', weights_only=True))
    return agent


def actor_batch(agent, observations):
    with torch.no_grad():
        return agent.actor(torch.as_tensor(observations, dtype=torch.float32)).numpy()


def candidates(source_action, updated_action, position, target):
    goal = np.asarray(target)[:2] - np.asarray(position)[:2]
    goal = goal / max(np.linalg.norm(goal), 1e-9)
    if np.linalg.norm(goal) < 1e-6:
        goal = np.array([1., 0.])
    theta = np.arctan2(goal[1], goal[0])
    values = [source_action, updated_action, .5 * (source_action + updated_action), np.zeros(3)]
    names = ['source', 'updated', 'midpoint', 'stop']
    for angle in np.arange(8) * np.pi / 4:
        direction = np.array([np.cos(theta + angle), np.sin(theta + angle)])
        for speed in (.25, .5, 1.):
            for z in (-1., 0., 1.):
                values.append(np.r_[speed * direction, z])
                names.append(f'heading{int(np.degrees(angle))}_norm{speed}_z{z}')
    actions = np.asarray(values, dtype=np.float32)
    assert np.all(np.linalg.norm(actions[:, :2], axis=1) <= 1 + 1e-6)
    assert np.max(np.abs(actions[:, 2])) <= 1 + 1e-6
    return actions, names


def q_table(agent, observation, actions):
    q1, q2 = agent.evaluate_critic_actions(observation, actions)
    assert np.all(np.isfinite(q1)) and np.all(np.isfinite(q2))
    return np.column_stack((q1, q2, np.minimum(q1, q2)))


def weight_delta(before, after):
    result = {}
    for name, predicate in (
        ('all', lambda key: True), ('gru', lambda key: key.startswith('encoder.gru.')),
        ('policy_mlp', lambda key: key.startswith('policy.')),
        ('output_layer', lambda key: key.startswith('policy.4.')),
    ):
        keys = [key for key in before if predicate(key)]
        old = np.concatenate([before[key].detach().numpy().ravel() for key in keys]).astype(np.float64)
        new = np.concatenate([after[key].detach().numpy().ravel() for key in keys]).astype(np.float64)
        result[name] = {'delta_l2': float(np.linalg.norm(new - old)),
                        'relative_delta_l2': float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12)),
                        'maximum_absolute_parameter_change': float(np.max(np.abs(new - old)))}
    return result


def command_metrics(source_action, updated_action, positions, target):
    source, updated = source_action * SCALE, updated_action * SCALE
    norm1, norm2 = np.linalg.norm(source[:, :2], axis=1), np.linalg.norm(updated[:, :2], axis=1)
    valid = (norm1 > .05) & (norm2 > .05)
    cos = np.sum(source[:, :2] * updated[:, :2], axis=1) / np.maximum(norm1 * norm2, 1e-12)
    angles = np.degrees(np.arccos(np.clip(cos[valid], -1, 1)))
    direction = target - positions
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    radial_source = np.sum(source * direction, axis=1)
    radial_updated = np.sum(updated * direction, axis=1)
    return {
        'states': len(source_action),
        'mean_command_difference_mps': float(np.mean(np.linalg.norm(updated - source, axis=1))),
        'mean_planar_heading_difference_degrees': float(np.mean(angles)) if len(angles) else None,
        'planar_heading_valid_states': int(valid.sum()),
        'planar_heading_over_90_degrees_fraction': float(np.mean(angles > 90)) if len(angles) else None,
        'source_mean_goal_closing_command_mps': float(radial_source.mean()),
        'updated_mean_goal_closing_command_mps': float(radial_updated.mean()),
        'source_away_command_fraction': float(np.mean(radial_source < -.02)),
        'updated_away_command_fraction': float(np.mean(radial_updated < -.02)),
        'mean_absolute_vertical_command_change_mps': float(np.mean(np.abs(updated[:, 2] - source[:, 2]))),
        'source_mean_vertical_command_mps': float(source[:, 2].mean()),
        'updated_mean_vertical_command_mps': float(updated[:, 2].mean()),
        'source_vertical_action_saturated_fraction': float(np.mean(np.abs(source_action[:, 2]) > .95)),
        'updated_vertical_action_saturated_fraction': float(np.mean(np.abs(updated_action[:, 2]) > .95)),
    }


def warmup_bank(seed):
    observations, positions, stages, distances, identifiers = [], [], [], [], []
    for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json')):
        meta = json.loads(path.read_text())
        if meta['global_upper_step'] > 4000:
            break
        if not meta['success']:
            continue
        with np.load(path.with_suffix('.npz')) as trace:
            obs = trace['upper_pre_observation'].copy()
            pos = np.vstack((meta['initial_position'], trace['upper_position'][:-1]))
        distance = np.linalg.norm(np.asarray(meta['target_position']) - pos, axis=1)
        observations.extend(obs); positions.extend(pos)
        stages.extend([meta['curriculum_stage']] * len(obs)); distances.extend(distance)
        identifiers.extend([f'{meta["trace_id"]}_step{i:03d}' for i in range(len(obs))])
    observations, positions = np.asarray(observations), np.asarray(positions)
    stages, distances = np.asarray(stages), np.asarray(distances)
    rng = np.random.default_rng(seed + 20261012)
    chosen = []
    for stage in (0, 3):
        for low, high in ((0., 1.2), (1.2, 4.), (4., 100.)):
            eligible = np.flatnonzero((stages == stage) & (distances >= low) & (distances < high))
            chosen.extend(rng.choice(eligible, min(32, len(eligible)), replace=False))
    chosen = np.asarray(sorted(chosen))
    assert len(chosen) > 0
    return {
        'observations': observations[chosen], 'positions': positions[chosen],
        'stages': stages[chosen], 'distances': distances[chosen],
        'identifiers': [identifiers[i] for i in chosen], 'eligible_success_states': len(observations),
        'target': np.array([3.5, 0., 1.]),
    }


def offline(seed, output):
    bank = warmup_bank(seed)
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    source_state = source.actor.state_dict()
    source_actions = actor_batch(source, bank['observations'])
    result = {'training_seed': seed, 'warmup_bank_size': len(source_actions),
              'eligible_success_states': bank['eligible_success_states'],
              'sampling': 'up to 32 successful warmup states per Stage 0/3 x 3D distance bin',
              'checkpoint_metrics': {}, 'early_online_output_drift': {}}
    q_records = []
    previous_state = source_state
    for step in CHECKPOINT_STEPS:
        prefix = Path(f'checkpoints/full_ft_v1_seed{seed}/model_{step}')
        agent = new_agent(prefix)
        updated_actions = actor_batch(agent, bank['observations'])
        actor_state = agent.actor.state_dict()
        record = {
            'actor_sha256': sha(str(prefix) + '_actor'), 'critic_sha256': sha(str(prefix) + '_critic'),
            'weight_change_from_source': weight_delta(source_state, actor_state),
            'weight_change_from_previous_scanned_checkpoint': weight_delta(previous_state, actor_state),
            'commands_on_fixed_success_bank': command_metrics(source_actions, updated_actions, bank['positions'], bank['target']),
            'q_ranking': {},
        }
        for subset_name, mask in (
            ('all', np.ones(len(source_actions), dtype=bool)),
            ('stage0', bank['stages'] == 0), ('stage3', bank['stages'] == 3),
            ('near_goal_1p2', bank['distances'] < 1.2),
        ):
            metrics = command_metrics(source_actions[mask], updated_actions[mask], bank['positions'][mask], bank['target']) if mask.any() else None
            if subset_name != 'all':
                record.setdefault('command_subsets', {})[subset_name] = metrics
        margins, spans, choices = [], [], []
        for i, observation in enumerate(bank['observations']):
            actions, names = candidates(source_actions[i], updated_actions[i], bank['positions'][i], bank['target'])
            values = q_table(agent, observation, actions)
            margins.append(values[1] - values[0]); spans.append(np.ptp(values, axis=0)); choices.append(int(np.argmax(values[:, 0])))
            q_records.append({
                'checkpoint_step': step, 'case_id': bank['identifiers'][i],
                'stage': int(bank['stages'][i]), 'goal_distance_m': float(bank['distances'][i]),
                'source_action': source_actions[i].tolist(), 'updated_action': updated_actions[i].tolist(),
                'source_q1_q2_min': values[0].tolist(), 'updated_q1_q2_min': values[1].tolist(),
                'best_q1_candidate': names[int(np.argmax(values[:, 0]))],
                'best_q1_action': actions[int(np.argmax(values[:, 0]))].tolist(),
                'best_q1_q1_q2_min': values[np.argmax(values[:, 0])].tolist(),
            })
        margins, spans = np.asarray(margins), np.asarray(spans)
        for col, name in enumerate(('q1', 'q2', 'q_min')):
            record['q_ranking'][name] = {
                'updated_preferred_fraction': float(np.mean(margins[:, col] > 1e-3)),
                'source_preferred_fraction': float(np.mean(margins[:, col] < -1e-3)),
                'mean_updated_minus_source_q': float(margins[:, col].mean()),
                'mean_candidate_q_span': float(spans[:, col].mean()),
            }
        record['q_ranking']['q1_q2_pair_order_disagreement_fraction'] = float(np.mean((margins[:, 0] > 1e-3) != (margins[:, 1] > 1e-3)))
        state_tensor = torch.as_tensor(bank['observations'], dtype=torch.float32)
        source_tensor = torch.as_tensor(source_actions, dtype=torch.float32).requires_grad_(True)
        value = agent.critic.Q1(state_tensor, source_tensor)
        action_gradient = torch.autograd.grad(value.sum(), source_tensor)[0].detach().numpy()
        drift = updated_actions - source_actions
        norms = np.linalg.norm(action_gradient, axis=1) * np.linalg.norm(drift, axis=1)
        valid = norms > 1e-8
        cosine = np.sum(action_gradient * drift, axis=1)[valid] / norms[valid]
        record['q1_action_gradient_at_source_action'] = {
            'mean_action_gradient_norm': float(np.mean(np.linalg.norm(action_gradient, axis=1))),
            'drift_aligned_with_positive_gradient_fraction': float(np.mean(cosine > 0)) if len(cosine) else None,
            'mean_drift_gradient_cosine': float(np.mean(cosine)) if len(cosine) else None,
            'scope': 'Gradient of the scanned checkpoint critic, not the unavailable 4000-step critic',
        }
        result['checkpoint_metrics'][str(step)] = record
        previous_state = actor_state
    early_rows = []
    for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json')):
        meta = json.loads(path.read_text()); start = meta['global_upper_step'] - meta['upper_steps'] + 1
        if start > 5000:
            break
        with np.load(path.with_suffix('.npz')) as trace:
            obs, logged = trace['upper_pre_observation'], trace['upper_policy_action']
        expected = actor_batch(source, obs)
        for i, (before, after) in enumerate(zip(expected, logged)):
            global_step = start + i
            if 3950 <= global_step <= 5000:
                early_rows.append({'upper_step': global_step,
                    'maximum_axis_action_difference': float(np.max(np.abs(before - after))),
                    'command_difference_mps': float(np.linalg.norm((before - after) * SCALE)),
                    'source_action': before.tolist(), 'logged_policy_action': after.tolist()})
    result['early_online_output_drift'] = {
        'first_changed_step': next(row['upper_step'] for row in early_rows if row['maximum_axis_action_difference'] > 1e-5),
        'selected_steps': [row for row in early_rows if row['upper_step'] in (4000, 4001, 4002, 4004, 4010, 4020, 4050, 4100, 4250, 4500, 5000)],
    }
    np.savez_compressed(output / f'seed{seed}_warmup_bank.npz',
        observations=bank['observations'], positions=bank['positions'], stages=bank['stages'], distances=bank['distances'])
    dump(output / f'seed{seed}_offline.json', result)
    dump(output / f'seed{seed}_offline_q_scan.json', q_records)
    dump(output / f'seed{seed}_early_online_drift.json', early_rows)
    print(f'Offline seed {seed}: {len(source_actions)} fixed successful Full states', flush=True)
    return result


def make_full_env():
    args = preset_args(BASE)
    args.map_id = 'procedural_forest_multitree_validation_random_v1'
    args.upper_execution_model = 'full'
    def factory(**kwargs):
        for key in ('kinematic_model', 'kinematic_planar_acceleration', 'kinematic_vertical_acceleration'):
            kwargs.pop(key)
        # The reused feasibility factory was written for K1, whose reward
        # rates integrate once per upper transition. Full integrates each
        # lower tick; preserve the production 1/120 s contract explicitly.
        kwargs['reward_model'].integration_dt = 1. / args.ctrl_freq
        return LearnedUpperDSLPIDForestEnv(**kwargs)
    return make_stage0_env(args, env_class=factory), args


def discounted_return(rewards, gamma):
    return float(sum(gamma ** i * float(reward) for i, reward in enumerate(rewards)))


def prefix_bootstrap(rewards, cut, gamma, tail_q):
    return discounted_return(rewards[:cut], gamma) + gamma ** cut * np.asarray(tail_q)


def replay_to_case(env, case, tape, metadata):
    env.set_curriculum_stage_override(case['stage'])
    obs, reset_info = env.reset(seed=case['map_seed'])
    assert reset_info['route_blocking_tree_count'] == (1 if case['stage'] == 0 else 5)
    assert abs(env.base_env._reward_model.integration_dt - 1. / 120) < 1e-12
    for index, action in enumerate(tape['upper_action'][:case['step']]):
        obs, reward, term, trunc, _ = env.step(action)
        assert not (term or trunc)
        np.testing.assert_allclose(reward, tape['upper_reward'][index], atol=1e-5, rtol=0)
    state = preprocess_state(obs.reshape(-1))
    np.testing.assert_allclose(state, tape['upper_pre_observation'][case['step']], atol=2e-5, rtol=0)
    pos = metadata['initial_position'] if case['step'] == 0 else tape['upper_position'][case['step'] - 1]
    vel = metadata['initial_velocity'] if case['step'] == 0 else tape['upper_velocity'][case['step'] - 1]
    np.testing.assert_allclose(env.pos[0], pos, atol=1e-6, rtol=0)
    np.testing.assert_allclose(env.vel[0], vel, atol=1e-6, rtol=0)
    assert env.runtime_lower_step == case['step'] * 8
    if case['step'] > 0:
        np.testing.assert_allclose(env.reference_velocity, tape['upper_reference_velocity'][case['step'] - 1], atol=1e-6, rtol=0)
        np.testing.assert_allclose(env.reference_position, tape['upper_reference_position'][case['step'] - 1], atol=1e-6, rtol=0)
    for actual, expected in zip(env._tree_specs, metadata['tree_specs']):
        np.testing.assert_allclose(actual['xy'], expected['xy'], atol=1e-6, rtol=0)
        assert abs(actual['radius'] - expected['radius']) < 1e-9
    assert len(env._tree_specs) == len(metadata['tree_specs'])
    return obs


def rollout_branch(env, source, updated, args, case, tape, metadata, kind, best_action):
    obs = replay_to_case(env, case, tape, metadata)
    horizon = 15 if kind == 'source15_then_updated' else 1
    continuation = source if kind in ('source_full', 'updated1_then_source') else updated
    forced = source if kind in ('source_full', 'source1_then_updated', 'source15_then_updated') else updated
    rewards, trace, q_tail = [], [], None
    q_tail15 = None
    discounted_reward_terms = {}
    first_action = None
    cut = None
    for k in range(220):
        state = preprocess_state(obs.reshape(-1))
        if kind == 'updated_full' and k == 15:
            q_tail15 = q_table(updated, state, updated.select_action(state)[None])[0]
        if k == horizon:
            continuation_action = continuation.select_action(state)
            q_tail = q_table(updated, state, np.asarray(continuation_action)[None])[0]
            cut = horizon
        action = forced.select_action(state) if k < horizon else continuation.select_action(state)
        if kind == 'best_q1_1_then_updated' and k == 0:
            action = best_action.copy()
        action = updated.project_action(action)
        if first_action is None:
            first_action = action.copy()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(float(reward) * args.reward_scale)
        for key, value in info.items():
            if key.startswith('reward/') and np.isscalar(value) and np.isfinite(value):
                discounted_reward_terms[key] = discounted_reward_terms.get(key, 0.) + args.gamma ** k * args.reward_scale * float(value)
        trace.append({'action': np.asarray(action).tolist(), 'reward_scaled': rewards[-1],
                      'goal_distance_m': float(info['goal_distance']),
                      'position': env.pos[0].tolist(), 'actual_velocity': env.vel[0].tolist(),
                      'command_velocity': env.command_velocity.tolist(),
                      'reference_velocity': env.reference_velocity.tolist(),
                      'clearance_m': (float(info['min_tree_clearance'])
                                      if np.isfinite(info['min_tree_clearance']) else None)})
        if term or trunc:
            if cut is None:
                cut = len(rewards); q_tail = np.zeros(3)
            break
    else:
        raise AssertionError('Branch did not end under the original 12-second limit')
    initial_q = q_table(updated, tape['upper_pre_observation'][case['step']], first_action[None])[0]
    row = {
        'case_id': case['case_id'], 'kind': kind, 'training_seed': case['training_seed'],
        'stage': case['stage'], 'map_seed': case['map_seed'], 'branch_step': case['step'],
        'forced_prefix_steps': cut, 'continuation_policy': 'source' if continuation is source else 'updated_5000',
        'first_action': first_action.tolist(), 'initial_q1_q2_min': initial_q.tolist(),
        'discounted_scaled_return': discounted_return(rewards, args.gamma),
        'prefix_plus_bootstrap_q1_q2_min': prefix_bootstrap(rewards, cut, args.gamma, q_tail).tolist(),
        'bootstrap_q1_q2_min': np.asarray(q_tail).tolist(),
        'outcome': info['done_reason'], 'remaining_upper_steps': len(rewards),
        'final_goal_distance_m': float(info['goal_distance']),
        'minimum_goal_distance_m': min(row['goal_distance_m'] for row in trace),
        'minimum_clearance_m': min((row['clearance_m'] for row in trace
                                   if row['clearance_m'] is not None), default=None),
        'reward_integration_dt_verified': float(env.base_env._reward_model.integration_dt),
        'discounted_scaled_reward_terms': discounted_reward_terms,
        'trace': trace,
    }
    if kind == 'updated_full':
        row['updated_full_15step_estimate'] = prefix_bootstrap(
            rewards, min(15, len(rewards)), args.gamma,
            np.zeros(3) if q_tail15 is None else q_tail15,
        ).tolist()
    if kind == 'source_full':
        row['cached_source_baseline_comparison'] = {
            'cached_outcome': metadata['done_reason'], 'replayed_outcome': row['outcome'],
            'upper_step_difference': len(rewards) - (len(tape['upper_action']) - case['step']),
            'final_goal_distance_difference_m': row['final_goal_distance_m'] - metadata['final_goal_distance'],
            'note': 'Matched initial state is strict; subsequent online floating-point policy feedback can change terminal timing. Actual comparisons use this newly executed baseline.',
        }
    return row


def branches(seed, output):
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    updated = new_agent(Path(f'checkpoints/full_ft_v1_seed{seed}/model_5000'))
    env, args = make_full_env()
    protocol = {'training_seed': seed, 'source_actor_sha256': sha(SOURCE / f'seed{seed}' / 'model_final_actor'),
                'critic_sha256': sha(f'checkpoints/full_ft_v1_seed{seed}/model_5000_critic'),
                'updated_actor_sha256': sha(f'checkpoints/full_ft_v1_seed{seed}/model_5000_actor'),
                'map_seeds': list(DIAGNOSTIC_MAP_SEEDS), 'stages': [0, 3],
                'branch_indices': '0, 60, first 3D guidance entry', 'gamma': args.gamma, 'reward_scale': args.reward_scale,
                'execution_projection': True, 'torch_num_threads': 1,
                'reward_integration_dt': 1. / 120}
    protocol_path = output / f'seed{seed}_branch_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        dump(protocol_path, protocol)
    rows_path, cases_path = output / f'seed{seed}_branches.json', output / f'seed{seed}_branch_cases.json'
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    complete_ids = {row['case_id'] for row in rows}
    cases = ([case for case in json.loads(cases_path.read_text()) if case['case_id'] in complete_ids]
             if cases_path.exists() else [])
    try:
        for stage in (0, 3):
            for map_seed in DIAGNOSTIC_MAP_SEEDS:
                folder = Path(f'runs/full_ft_v1_seed{seed}/full_stage{stage}_traces/step_0000000')
                with np.load(folder / f'trace_seed{map_seed}.npz') as data:
                    tape = {key: data[key].copy() for key in data.files}
                metadata = json.loads((folder / f'trace_seed{map_seed}.json').read_text())
                assert metadata['success'], 'Diagnostic maps must be successful source trajectories'
                positions = np.vstack((metadata['initial_position'], tape['upper_position'][:-1]))
                distances = np.linalg.norm(np.asarray(metadata['target_position']) - positions, axis=1)
                guidance = np.flatnonzero(distances < 1.2)
                assert len(guidance), 'Successful source trajectory did not enter 1.2m ball'
                for index in dict.fromkeys((0, 60, int(guidance[0]))):
                    case = {'case_id': f'seed{seed}_stage{stage}_map{map_seed}_step{index:03d}',
                            'training_seed': seed, 'stage': stage, 'map_seed': map_seed, 'step': index,
                            'goal_distance_m': float(distances[index]),
                            'remaining_nominal_seconds': 12 - index / 15}
                    if case['case_id'] in complete_ids:
                        assert len([row for row in rows if row['case_id'] == case['case_id']]) == 6
                        continue
                    obs = replay_to_case(env, case, tape, metadata)
                    state = preprocess_state(obs.reshape(-1))
                    a_source, a_updated = source.select_action(state), updated.select_action(state)
                    action_grid, names = candidates(a_source, a_updated, env.pos[0], env.TARGET_POS)
                    values = q_table(updated, state, action_grid)
                    best_index = int(np.argmax(values[:, 0]))
                    case.update({'source_action': a_source.tolist(), 'updated_action': a_updated.tolist(),
                                 'source_q1_q2_min': values[0].tolist(), 'updated_q1_q2_min': values[1].tolist(),
                                 'best_q1_candidate': names[best_index], 'best_q1_action': action_grid[best_index].tolist(),
                                 'candidate_q1_q2_min': values.tolist(), 'candidate_names': names})
                    cases.append(case)
                    for kind in ('source_full', 'updated_full', 'source1_then_updated',
                                 'updated1_then_source', 'source15_then_updated', 'best_q1_1_then_updated'):
                        rows.append(rollout_branch(env, source, updated, args, case, tape, metadata, kind, action_grid[best_index]))
                    dump(output / f'seed{seed}_branch_cases.json', cases)
                    dump(output / f'seed{seed}_branches.json', rows)
                    print(f'{case["case_id"]}: 6 matched-state branches completed', flush=True)
    finally:
        env.close()
    return rows


def regrade(seed, output):
    """Re-execute saved branch actions under the checked Full reward contract.

    Rewards never enter the observation or frozen policy. Prefix, position,
    velocity and termination assertions verify that this fixed-action replay
    is the same branch, while all rewards and bootstraps are recalculated.
    """
    rows_path = output / f'seed{seed}_branches.json'
    rows = json.loads(rows_path.read_text())
    cases = {case['case_id']: case for case in json.loads((output / f'seed{seed}_branch_cases.json').read_text())}
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    updated = new_agent(Path(f'checkpoints/full_ft_v1_seed{seed}/model_5000'))
    env, args = make_full_env()
    try:
        for j, row in enumerate(rows):
            case = cases[row['case_id']]
            folder = Path(f'runs/full_ft_v1_seed{seed}/full_stage{case["stage"]}_traces/step_0000000')
            with np.load(folder / f'trace_seed{case["map_seed"]}.npz') as data:
                tape = {key: data[key].copy() for key in data.files}
            metadata = json.loads((folder / f'trace_seed{case["map_seed"]}.json').read_text())
            obs = replay_to_case(env, case, tape, metadata)
            rewards, terms = [], {}
            tail, tail15 = np.zeros(3), np.zeros(3)
            continuation = source if row['continuation_policy'] == 'source' else updated
            for k, sample in enumerate(row['trace']):
                if k == row['forced_prefix_steps']:
                    state = preprocess_state(obs.reshape(-1))
                    tail = q_table(updated, state, continuation.select_action(state)[None])[0]
                if k == 15 and row['kind'] == 'updated_full':
                    state = preprocess_state(obs.reshape(-1))
                    tail15 = q_table(updated, state, updated.select_action(state)[None])[0]
                obs, reward, terminated, truncated, info = env.step(np.asarray(sample['action'], dtype=np.float32))
                np.testing.assert_allclose(env.pos[0], sample['position'], atol=1e-6, rtol=0)
                np.testing.assert_allclose(env.vel[0], sample['actual_velocity'], atol=1e-6, rtol=0)
                assert bool(terminated or truncated) == (k == len(row['trace']) - 1)
                rewards.append(float(reward) * args.reward_scale)
                sample['reward_scaled'] = rewards[-1]
                for key, value in info.items():
                    if key.startswith('reward/') and np.isscalar(value) and np.isfinite(value):
                        terms[key] = terms.get(key, 0.) + args.gamma ** k * args.reward_scale * float(value)
            assert info['done_reason'] == row['outcome']
            row['discounted_scaled_return'] = discounted_return(rewards, args.gamma)
            row['prefix_plus_bootstrap_q1_q2_min'] = prefix_bootstrap(rewards, row['forced_prefix_steps'], args.gamma, tail).tolist()
            if row['kind'] == 'updated_full':
                row['updated_full_15step_estimate'] = prefix_bootstrap(rewards, min(15, len(rewards)), args.gamma, tail15).tolist()
            row['discounted_scaled_reward_terms'] = terms
            row['reward_integration_dt_verified'] = 1. / 120
            # Full reward identity: progress - continuous cost + terminal outcome.
            identity = terms['reward/progress_reward'] - terms['reward/continuous_penalty_total'] + terms['reward/terminal_outcome_reward']
            np.testing.assert_allclose(row['discounted_scaled_return'], identity, atol=1e-5, rtol=0)
            if (j + 1) % 6 == 0:
                print(f'Regraded seed {seed}: {j + 1}/{len(rows)} Full branches', flush=True)
        dump(rows_path, rows)
        protocol_path = output / f'seed{seed}_branch_protocol.json'
        protocol = json.loads(protocol_path.read_text()); protocol['reward_integration_dt'] = 1. / 120
        dump(protocol_path, protocol)
    finally:
        env.close()


def ranking_summary(rows, cases):
    paired = []
    for case in cases:
        selected = {row['kind']: row for row in rows if row['case_id'] == case['case_id']}
        a, b = selected['source1_then_updated'], selected['updated_full']
        actual_margin = b['discounted_scaled_return'] - a['discounted_scaled_return']
        value_margin = np.asarray(b['initial_q1_q2_min']) - a['initial_q1_q2_min']
        safe_margin = selected['updated1_then_source']['discounted_scaled_return'] - selected['source_full']['discounted_scaled_return']
        prefix_a = selected['source15_then_updated']
        # The updated full policy supplies the common 15-step comparison prefix.
        updated15_estimate = b['updated_full_15step_estimate']
        fifteen_actual = b['discounted_scaled_return'] - prefix_a['discounted_scaled_return']
        fifteen_estimate = np.asarray(updated15_estimate) - prefix_a['prefix_plus_bootstrap_q1_q2_min']
        paired.append({'case_id': case['case_id'], 'stage': case['stage'],
            'goal_distance_m': case['goal_distance_m'], 'actual_updated_minus_source_one_step': actual_margin,
            'q_updated_minus_source_one_step_q1_q2_min': value_margin.tolist(),
            'actual_updated_minus_source_one_step_with_source_continuation': safe_margin,
            'actual_updated_full_minus_source_full': b['discounted_scaled_return'] - selected['source_full']['discounted_scaled_return'],
            'actual_updated_minus_source_15step': fifteen_actual,
            'estimate_updated_minus_source_15step_q1_q2_min': fifteen_estimate.tolist(),
            'outcomes': {key: item['outcome'] for key, item in selected.items()}})
    result = {'cases': len(cases), 'rollouts': len(rows), 'paired_rows': paired,
              'source_full_outcomes': dict(Counter(row['outcomes']['source_full'] for row in paired)),
              'updated_full_outcomes': dict(Counter(row['outcomes']['updated_full'] for row in paired)),
              'one_step_common_updated_continuation': {}, 'fifteen_step_common_updated_continuation': {}}
    for prefix, actual_key, estimate_key in (
        ('one_step_common_updated_continuation', 'actual_updated_minus_source_one_step', 'q_updated_minus_source_one_step_q1_q2_min'),
        ('fifteen_step_common_updated_continuation', 'actual_updated_minus_source_15step', 'estimate_updated_minus_source_15step_q1_q2_min'),
    ):
        actual = np.asarray([row[actual_key] for row in paired])
        estimate = np.asarray([row[estimate_key] for row in paired])
        result[prefix]['mean_actual_updated_minus_source_return'] = float(actual.mean())
        for col, name in enumerate(('q1', 'q2', 'q_min')):
            informative = np.abs(actual) > .05
            decisive = informative & (np.abs(estimate[:, col]) > 1e-3)
            reversed_order = decisive & (actual * estimate[:, col] < 0)
            result[prefix][name] = {'informative_pairs': int(informative.sum()),
                'decisive_pairs': int(decisive.sum()), 'reversed_order_pairs': int(reversed_order.sum()),
                'reversed_order_fraction': float(reversed_order.sum() / decisive.sum()) if decisive.any() else None,
                'updated_preferred_count': int(np.sum(estimate[:, col] > 1e-3)),
                'mean_estimated_updated_minus_source': float(estimate[:, col].mean())}
    result['one_action_calibration'] = {}
    for kind in ('updated_full', 'source1_then_updated', 'best_q1_1_then_updated'):
        selection = [row for row in rows if row['kind'] == kind]
        values = np.asarray([row['initial_q1_q2_min'] for row in selection])
        actual_values = np.asarray([row['discounted_scaled_return'] for row in selection])
        result['one_action_calibration'][kind] = {'rollouts': len(selection),
            'mean_actual_discounted_scaled_return': float(actual_values.mean()),
            'outcomes': dict(Counter(row['outcome'] for row in selection)),
            'mean_q1_q2_min': values.mean(axis=0).tolist(),
            'mean_q_minus_actual_q1_q2_min': (values - actual_values[:, None]).mean(axis=0).tolist(),
            'mean_abs_q_minus_actual_q1_q2_min': np.abs(values - actual_values[:, None]).mean(axis=0).tolist()}
    baseline = [row for row in rows if row['kind'] == 'source_full']
    result['source_full_mean_discounted_scaled_return'] = float(np.mean([row['discounted_scaled_return'] for row in baseline]))
    result['source_baseline_cached_outcome_disagreements'] = sum(
        row.get('cached_source_baseline_comparison', {}).get('cached_outcome', row['outcome']) != row['outcome'] for row in baseline)
    result['source_baseline_cached_timing_disagreements'] = sum(
        row.get('cached_source_baseline_comparison', {}).get('upper_step_difference', 0) != 0 for row in baseline)
    return result


def reward_probe(seed, output):
    source = new_agent(SOURCE / f'seed{seed}' / 'model_final')
    updated = new_agent(Path(f'checkpoints/full_ft_v1_seed{seed}/model_5000'))
    cases = json.loads((output / f'seed{seed}_branch_cases.json').read_text())
    case = next(case for case in cases if case['map_seed'] == 80 and case['stage'] == (3 if seed == 43 else 0) and case['step'] > 60)
    folder = Path(f'runs/full_ft_v1_seed{seed}/full_stage{case["stage"]}_traces/step_0000000')
    with np.load(folder / 'trace_seed80.npz') as data:
        tape = {key: data[key].copy() for key in data.files}
    metadata = json.loads((folder / 'trace_seed80.json').read_text())
    env, args = make_full_env()
    try:
        rows = [rollout_branch(env, source, updated, args, case, tape, metadata, kind,
                              np.asarray(case['best_q1_action'], dtype=np.float32))
                for kind in ('source_full', 'updated_full', 'source15_then_updated')]
        dump(output / f'seed{seed}_near_goal_reward_probe.json', rows)
        print(f'Reward probe seed {seed}: {case["case_id"]}', flush=True)
    finally:
        env.close()


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, seed in zip(axes, (43, 45)):
        path = output / f'seed{seed}_early_online_drift.json'
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        x = np.asarray([row['upper_step'] - 4000 for row in rows])
        y = np.asarray([row['command_difference_mps'] for row in rows])
        ax.plot(x, y, alpha=.3, linewidth=.7, label='Per-state command difference')
        smoothed = np.convolve(y, np.ones(25) / 25, mode='valid')
        ax.plot(x[24:], smoothed, linewidth=2, label='25-step trailing mean')
        first = next(row for row in rows if row['upper_step'] == 4002)
        ax.scatter([2], [first['command_difference_mps']], c='red', zorder=5)
        ax.annotate(f'First update: {first["command_difference_mps"]:.3f} m/s',
                    xy=(2, first['command_difference_mps']), xytext=(100, .08),
                    arrowprops={'arrowstyle': '->'}, fontsize=9)
        ax.axvspan(-50, 0, color='gray', alpha=.15)
        ax.set(title=f'Training seed {seed}', xlabel='Upper steps relative to Actor unfreeze', xlim=(-50, 1000))
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('3D command difference from source (m/s)')
    fig.suptitle('Actor output drift on identical recorded Full observations')
    fig.tight_layout(); fig.savefig(output / 'actor_update_amplitude.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row_index, seed in enumerate((43, 45)):
        if str(seed) not in summary['seeds']:
            continue
        calibration = summary['seeds'][str(seed)]['calibration']
        pairs = calibration['paired_rows']
        branches_data = json.loads((output / f'seed{seed}_branches.json').read_text())
        for column, (actual_key, estimate_key, title) in enumerate((
            ('actual_updated_minus_source_one_step', 'q_updated_minus_source_one_step_q1_q2_min', 'One action + common updated policy'),
            ('actual_updated_minus_source_15step', 'estimate_updated_minus_source_15step_q1_q2_min', '15-step prefix + common updated policy'),
        )):
            ax = axes[row_index, column]
            actual = np.asarray([pair[actual_key] for pair in pairs])
            predicted = np.asarray([pair[estimate_key][0] for pair in pairs])
            for stage, color in ((0, '#1f77b4'), (3, '#ff7f0e')):
                indices = np.asarray([pair['stage'] == stage for pair in pairs])
                ax.scatter(actual[indices], predicted[indices], color=color, label=f'Stage {stage}', s=40)
            lower = min(actual.min(), predicted.min(), 0); upper = max(actual.max(), predicted.max(), 0)
            pad = max((upper - lower) * .1, .1)
            ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], '--', color='gray', linewidth=.8)
            ax.axhline(0, color='gray', linewidth=.5); ax.axvline(0, color='gray', linewidth=.5)
            ax.set(title=f'Seed {seed}: {title}', xlabel='Actual return difference', ylabel='Estimated difference (Q1)',
                   xlim=(lower - pad, upper + pad), ylim=(lower - pad, upper + pad))
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        ax = axes[row_index, 2]
        selection = [row for row in branches_data if row['kind'] == 'updated_full']
        actual = np.asarray([row['discounted_scaled_return'] for row in selection])
        predicted = np.asarray([row['initial_q1_q2_min'][0] for row in selection])
        for stage, color in ((0, '#1f77b4'), (3, '#ff7f0e')):
            indices = np.asarray([row['stage'] == stage for row in selection])
            ax.scatter(actual[indices], predicted[indices], color=color, label=f'Stage {stage}', s=40)
        lower = min(actual.min(), predicted.min(), 0); upper = max(actual.max(), predicted.max(), 0)
        pad = max((upper - lower) * .1, .1)
        ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], '--', color='gray', linewidth=.8)
        ax.axhline(0, color='gray', linewidth=.5); ax.axvline(0, color='gray', linewidth=.5)
        ax.set(title=f'Seed {seed}: updated-policy return calibration', xlabel='Actual discounted scaled return',
               ylabel='Initial Q1', xlim=(lower - pad, upper + pad), ylim=(lower - pad, upper + pad))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Full critic audit: gamma=0.99, reward scale=0.1, lower reward dt=1/120 s')
    fig.tight_layout(); fig.savefig(output / 'critic_return_calibration.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('offline', 'branches', 'regrade', 'reward_probe', 'summary', 'all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=(43, 45), choices=(43, 45))
    args = parser.parse_args()
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for seed in args.seeds:
        source_actor = SOURCE / f'seed{seed}' / 'model_final_actor'
        assert sha(source_actor) == manifest['seeds'][str(seed)]['files']['model_final_actor']['sha256']
    if args.phase in ('all', 'offline'):
        for seed in args.seeds:
            offline(seed, ROOT)
    if args.phase in ('all', 'branches'):
        for seed in args.seeds:
            branches(seed, ROOT)
    if args.phase == 'regrade':
        for seed in args.seeds:
            regrade(seed, ROOT)
    if args.phase in ('all', 'reward_probe'):
        for seed in args.seeds:
            reward_probe(seed, ROOT)
    if args.phase in ('all', 'summary'):
        summary = {'metadata': {'critic_checkpoint_step': 5000,
            'missing_preunfreeze_critic_checkpoint': True, 'gamma': .99, 'reward_scale': .1,
            'full_reward_integration_dt': 1. / 120,
            'diagnostic_environment_seeds': list(DIAGNOSTIC_MAP_SEEDS),
            'heldout_40_79_used': False, 'no_training_or_optimizer_updates': True,
            'one_step_protocol': 'One candidate action, then common online Actor at 5000 until original termination',
            'fifteen_step_protocol': '15 forced source-policy actions, then common online Actor at 5000; compare prefix reward + discounted bootstrap, never initial Q',
            'limitations': 'Critic 5000 is already after 500 Actor updates; original target-Actor lag, target smoothing and hidden remaining clock differ from diagnostic deterministic online continuation.'}, 'seeds': {}}
        for seed in args.seeds:
            off = json.loads((ROOT / f'seed{seed}_offline.json').read_text())
            rows = json.loads((ROOT / f'seed{seed}_branches.json').read_text())
            cases = json.loads((ROOT / f'seed{seed}_branch_cases.json').read_text())
            assert len(rows) == 72 and len(cases) == 12
            assert all(abs(row.get('reward_integration_dt_verified', 0.) - 1. / 120) < 1e-12 for row in rows)
            calibration = ranking_summary(rows, cases)
            calibration['by_stage'] = {str(stage): ranking_summary(
                [row for row in rows if row['stage'] == stage], [case for case in cases if case['stage'] == stage])
                for stage in (0, 3)}
            summary['seeds'][str(seed)] = {'offline': off, 'calibration': calibration}
        dump(ROOT / 'summary.json', summary)
        plot_summary(summary, ROOT)
        print(ROOT / 'summary.json')


if __name__ == '__main__':
    main()
