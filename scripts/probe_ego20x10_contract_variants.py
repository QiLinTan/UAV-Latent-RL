"""Run single-factor engineering/physical probes, without learner training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_ego20x10_value import SOURCE, load, describe, relocate_reset, write
from scripts.audit_full_ego20x10_task_contract import (
    ROOT as M0_ROOT, raw_state, reward_on, free_space_env,
)
from scripts.run_full_critic_frozen import production_hashes
from scripts.run_full_ft_v2 import sha
from scripts.ego20x10_contract_variants import (
    CONTRACTS, WORLD_SPEED, attach_contract, contract_env, encode_observation,
    reencode_replay_state, desired_velocity, project_velocity, FeasibleGuidanceReward,
)

ROOT = Path('analysis_outputs/full_ego20x10_contract_corrections_v1')


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protected = production_hashes()
    protected.update({str(p): sha(p) for p in SOURCE.iterdir() if p.is_file()})
    old = Path('checkpoints/full_critic_frozen_v1_seed43')
    protected.update({str(p): sha(p) for p in old.iterdir() if p.is_file()})
    write(root/'protected_inputs.json', protected)
    write(root/'protocol.json', {
        'variants': [asdict(c) for c in CONTRACTS.values()], 'learner_updates': 0,
        'method': 'Single-factor state reencoding and identical Full action tapes; no source-policy comparison across incompatible encodings.',
        'fixed': {'state_dim': 917, 'success_radius': .2, 'guidance_radius': 1.2,
                  'episode_limit_s': 40, 'upper_hz': 15, 'lower_hz': 120,
                  'radar_m': 1.75, 'reward_scale': .1, 'courses': 'unchanged'},
        'test': 'No test spec, path, policy episode or tuning. Production constructor validates the unified pool manifest.',
        'source_6k_sha256': sha(SOURCE/'training_state_step6000.pt'),
    })


def encoding_and_rewards(root, saved):
    states = saved['replay']['arrays']['state']
    position = states[:, :3]*.5+[0., 0., 1.]
    statistics = {}
    for name, contract in CONTRACTS.items():
        converted = reencode_replay_state(states, contract)
        statistics[name] = {
            'position': describe(converted[:, :3]), 'goal': describe(converted[:, 12:15]),
            'same_history': bool(np.array_equal(converted[:, 117:], states[:, 117:])),
            'unchanged_other_current_fields': bool(np.array_equal(converted[:, 3:12], states[:, 3:12])
                and np.array_equal(converted[:, 15:117], states[:, 15:117])),
            'goal_x_saturated_fraction': float(np.mean(np.abs(converted[:, 12]) >= 1.)),
        }
        if contract.goal_range:
            np.testing.assert_allclose(converted[:, 12:15]*[20., 10., 2.], np.array([8., 0., 1.])-position, atol=2e-6)
        if contract.position_scale:
            np.testing.assert_allclose(converted[:, :3]*[10., 5., 1.]+[0., 0., 1.], position, atol=2e-6)
    write(root/'encoding.json', statistics)
    env = contract_env(saved, CONTRACTS['baseline'])
    try:
        old = env.base_env._reward_model
        new = FeasibleGuidanceReward(old)
        costs, changes, infeasible = [], [], []
        for s, p in zip(states, position):
            if np.linalg.norm(p-env.TARGET_POS) >= 1.2:
                continue
            state = raw_state(p, s[6:9]*2, s[3:6]*np.pi)
            a = reward_on(old, env, state, state)
            b = reward_on(new, env, state, state)
            costs.append([a[1]['near_goal_velocity_tracking_penalty']*120, b[1]['near_goal_velocity_tracking_penalty']*120])
            changes.append((b[0]-a[0])*120)
            infeasible.append(abs(desired_velocity(old, p, env.TARGET_POS)[2]) > .3)
        costs = np.asarray(costs)
        write(root/'guidance_replay_rates.json', {
            'states': len(costs), 'old_tracking_rate': describe(costs[:, 0]),
            'new_tracking_rate': describe(costs[:, 1]), 'reward_rate_change': describe(changes),
            'old_infeasible_fraction': float(np.mean(infeasible)),
            'new_target_feasible': True,
            'note': 'Instantaneous rates on real Replay states; not reaggregated episode rewards or a converted learning Replay.',
        })
    finally:
        env.close()
    references = json.loads((M0_ROOT/'M0C_full_40s_reference_budgets.json').read_text())
    discounts = {}
    for gamma in (.99, .997):
        discounts[str(gamma)] = {
            'half_life_s': float(np.log(.5)/np.log(gamma)/15),
            'geometric_horizon_s': float(1/(1-gamma)/15),
            'terminal_discount_20s': gamma**300, 'terminal_discount_40s': gamma**600,
            'reference_returns_scaled': {r['behavior']: float(sum(gamma**i*s['reward']*.1
                for i, s in enumerate(r['samples']))) for r in references},
        }
    write(root/'discount.json', discounts)


def physical_tapes(root, saved):
    """Each variant gets fresh resets, the same commands, and every lower tick."""
    results = {}
    for name, contract in CONTRACTS.items():
        env = attach_contract(free_space_env(saved), contract)
        records = []
        try:
            for task, start in (('initial16m', [-8., 0., 1.]),
                                ('near_above', [7.7, 0., 1.8]),
                                ('near_below', [7.7, 0., .4])):
                obs, _ = env.reset(seed=43)
                obs = relocate_reset(env, start)
                initial_encoded = encode_observation(obs, contract)
                assert initial_encoded.shape == (917,) and np.isfinite(initial_encoded).all()
                original = env.base_env.step
                ticks = []
                def lower(action):
                    output = original(action)
                    ticks.append((env.pos[0].copy(), env.vel[0].copy(), float(output[1])))
                    return output
                env.base_env.step = lower
                total, penalties, samples = 0., 0., []
                # Scripted controller is identical across all candidates. Goal and
                # position encodings are not used to generate commands in this probe.
                try:
                    for upper in range(60):
                        delta = env.TARGET_POS-env.pos[0]
                        speed = .8 if np.linalg.norm(delta) >= 1.2 else .1+.7*np.clip((np.linalg.norm(delta)-.2), 0., 1.)
                        velocity = project_velocity(speed*delta/max(float(np.linalg.norm(delta)), 1e-9))
                        action = (velocity/WORLD_SPEED).astype(np.float32)
                        obs, reward, term, trunc, info = env.step(action.reshape(1, 3))
                        encoded = encode_observation(obs, contract)
                        assert encoded.shape == (917,) and np.isfinite(encoded).all()
                        total += reward
                        penalties += info.get('reward/near_goal_velocity_tracking_penalty', 0.)
                        samples.append([*action, reward])
                        if term or trunc:
                            break
                finally:
                    env.base_env.step = original
                np.savez_compressed(root/f'{name}_{task}.npz',
                    initial_state=initial_encoded, lower_position=np.array([t[0] for t in ticks]),
                    lower_velocity=np.array([t[1] for t in ticks]), lower_reward=np.array([t[2] for t in ticks]),
                    upper_action=np.array(samples)[:, :3], upper_reward=np.array(samples)[:, 3])
                records.append({'task': task, 'raw_return': total, 'tracking_cost': penalties,
                    'lower_ticks': len(ticks), 'upper_steps': len(samples), 'outcome': info['done_reason'],
                    'final_distance': float(info['goal_distance']),
                    'max_upper_reward_sum_error': float(max(abs(r[3]-sum(t[2] for t in ticks[i*8:min((i+1)*8, len(ticks))])) for i, r in enumerate(samples)))})
                print(f'[Contract] {name} {task} ticks={len(ticks)} R={total:.6f}', flush=True)
            results[name] = records
        finally:
            env.close()
    for name in CONTRACTS:
        for record in results[name]:
            task = record['task']
            with np.load(root/f'baseline_{task}.npz') as a, np.load(root/f'{name}_{task}.npz') as b:
                for field in ('lower_position', 'lower_velocity', 'upper_action'):
                    np.testing.assert_array_equal(a[field], b[field])
                if name != 'guidance_project':
                    np.testing.assert_array_equal(a['lower_reward'], b['lower_reward'])
                record['physical_tape_matches_baseline'] = True
                if name == 'baseline':
                    record['reward_changed'] = False
                else:
                    record['reward_changed'] = bool(not np.array_equal(a['lower_reward'], b['lower_reward']))
    write(root/'physical_proof.json', results)
    # Also verify original forest instances (all four densities), not just an
    # empty rectangle. Same first action and reset map index in all variants.
    smoke = []
    for stage in range(4):
        reference = None
        for name, contract in CONTRACTS.items():
            env = contract_env(saved, contract)
            try:
                env.set_curriculum_stage_override(stage)
                obs, _ = env.reset(seed=0)
                old_position = env.pos[0].copy()
                obs, reward, term, trunc, info = env.step(np.array([[.25, -.1, .05]], dtype=np.float32))
                encoded = encode_observation(obs, contract)
                assert encoded.shape == (917,)
                actual = (env.pos[0].copy(), env.vel[0].copy(), reward)
                if reference is None:
                    reference = actual
                else:
                    for a, b in zip(reference, actual):
                        np.testing.assert_array_equal(a, b)
                smoke.append({'variant': name, 'stage': stage, 'trees': len(env.base_env._tree_ids),
                    'position_delta': (env.pos[0]-old_position).tolist(), 'lower_ticks': env.runtime_lower_step,
                    'state_dim': len(encoded), 'same_physics_and_reward': True,
                    'map_split': info['map_split'], 'map_id': info['map_id']})
            finally:
                env.close()
    write(root/'forest_smoke.json', smoke)


def finish(root):
    guard = json.loads((root/'protected_inputs.json').read_text())
    changed = [p for p, digest in guard.items() if sha(Path(p)) != digest]
    write(root/'protection_result.json', {'unchanged': not changed, 'changed': changed,
        'checked_files': len(guard), 'learner_updates': 0, 'test_episodes': 0})
    if changed:
        raise RuntimeError(f'Protected inputs changed: {changed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.output)
    saved = load(6000)
    encoding_and_rewards(args.output, saved)
    physical_tapes(args.output, saved)
    finish(args.output)


if __name__ == '__main__':
    main()
