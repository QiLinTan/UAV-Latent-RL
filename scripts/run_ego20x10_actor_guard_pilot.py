"""Short shared-Replay Actor-step ablation; not an online continuation.

Uses the original task and random 5k Actor. The 100-Actor-update experiment
tests whether a local output bound alone preserves its initial L0 behavior.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_ego20x10_value import (
    DEFAULT as BASELINE, SOURCE, load, learner, rng_restore, fingerprint,
    env_for, relocate_reset, write,
)
from scripts.run_ego20x10_d3 import (
    ScheduledBuffer, policy_probe, NETWORKS, save_checkpoint, checkpoint,
)
from scripts.evaluate_ego20x10_d3 import LowerCollector, eval_episode
from scripts.ego20x10_contract_variants import ActorStepGuard, WORLD_SPEED
from scripts.run_full_critic_frozen import production_hashes
from scripts.run_full_ft_v2 import sha

ROOT = Path('analysis_outputs/full_ego20x10_actor_guard_pilot_v1')
CHECKPOINT_COUNTS = (0, 1, 10, 50, 100)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    guard = production_hashes()
    guard.update({str(p): sha(p) for p in SOURCE.iterdir() if p.is_file()})
    guard.update({str(p): sha(p) for p in BASELINE.rglob('*') if p.is_file()})
    write(root/'protected_inputs.json', guard)
    write(root/'protocol.json', {
        'new_online_transitions': 0, 'new_upper_replay_items': 0,
        'critic_updates_per_branch': 200, 'actor_proposals_per_branch': 100,
        'branches': ['baseline', 'step_guard'],
        'mean_step_limit_mps': .02, 'p95_step_limit_mps': .05, 'backtracks': 10,
        'fixed_guard_states': '188 original 5k training Replay states; no validation states used by guard',
        'data': 'Exact recorded main/success indices and target noise from A5001..5200; each branch computes its own TD targets.',
        'actor_target': 'Standard Polyak for accepted proposal; rejected proposal restores target/Actor/Adam.',
        'task': 'Original16m contract, unchanged rewards/encoding/gamma/GRU/curriculum',
        'validation': '20 frozen Stage0 maps, paired L0 at actor proposals0/1/10/50/100; original16m at100.',
        'limits': 'Engineering step bounds are candidates, not established safe thresholds. Cumulative drift is unbounded.',
        'no_test': True, 'source_5k_sha256': sha(SOURCE/'training_state_step5000.pt'),
        'schedule_sha256': sha(BASELINE/'batch_schedule.npz'),
    })


def train(root):
    five, six = load(5000), load(6000)
    schedule = np.load(BASELINE/'batch_schedule.npz')
    fixed = np.load(BASELINE/'fixed_states.npz')['states']
    initial = learner(copy.deepcopy(five))
    with torch.no_grad():
        initial_action = initial.actor(torch.as_tensor(fixed)).numpy()
    models = {label: learner(copy.deepcopy(five)) for label in ('baseline', 'step_guard')}
    guard = ActorStepGuard(models['step_guard'], fixed)
    rng_restore(five)
    random_states = {label: torch.get_rng_state().clone() for label in models}
    records = {label: [] for label in models}
    probes = {label: [] for label in models}
    for label in models:
        (root/label/'checkpoints').mkdir(parents=True)
    def snapshot(label, step, proposals, accepted):
        agent = models[label]
        values, actions = policy_probe(agent, fixed)
        delta = np.linalg.norm((actions-initial_action)*WORLD_SPEED, axis=1)
        values.update(step=int(step), actor_proposals=proposals, actor_updates=accepted,
            cumulative_change_mean_mps=float(delta.mean()), cumulative_change_p95_mps=float(np.quantile(delta, .95)),
            actor_sha256=fingerprint(agent.actor), critic_sha256=fingerprint(agent.critic))
        probes[label].append(values)
        save_checkpoint(agent, root/label, int(step), accepted, five)
        write(root/label/'checkpoint_probes.json', probes[label])
        print(f'[Guard pilot] {label} proposal={proposals} accepted={accepted} az={values["action_mean"][2]:+.6f} sat={values["z_saturation_fraction"]:.1%} cumulative={delta.mean():.4f}m/s', flush=True)
    for label in models:
        snapshot(label, 5000, 0, 0)
    accepted_counts = {label: 0 for label in models}
    proposals = 0
    for n, step in enumerate(schedule['steps'][:200]):
        step = int(step)
        if n % 2 == 0:
            proposals += 1
        for label, agent in models.items():
            main = ScheduledBuffer(six['replay']['arrays'])
            main.set_batch(schedule['main_indices'][n], step)
            success = ScheduledBuffer(five['success']['strata'][10]['arrays'])
            success.set_batch(schedule['success_indices'][n], 297)
            torch.set_rng_state(random_states[label])
            generator = torch.Generator().set_state(random_states[label])
            expected_noise = (torch.randn((256, 3), generator=generator)*agent.policy_noise).clamp(-agent.noise_clip, agent.noise_clip)
            np.testing.assert_array_equal(expected_noise.numpy(), schedule['target_noise'][n])
            if label == 'step_guard':
                info, transaction = guard.train(main, 256, success, .2)
            else:
                info, transaction = agent.train(main, 256, success, .2), None
            random_states[label] = torch.get_rng_state().clone()
            torch.testing.assert_close(random_states[label], generator.get_state(), atol=0, rtol=0)
            accepted_counts[label] += int(info['actor_updated'])
            records[label].append({'step': step, 'actor_proposals': proposals, 'actor_updates': accepted_counts[label],
                                  'train': info, 'guard': transaction})
            # Save immediately after the matching Actor proposal, before its next
            # critic-only update. Always save exact step5200 for baseline equality.
            if (n % 2 == 0 and proposals in CHECKPOINT_COUNTS) or n == 199:
                snapshot(label, step, proposals, accepted_counts[label])
    for label in models:
        write(root/label/'updates.json', records[label])
    # The no-guard control must equal the previously replicated online A learner.
    comparisons = {}
    for name in ('actor', 'critic'):
        expected = torch.load(BASELINE/'checkpoints'/f'model_raw_5200_{name}', map_location='cpu', weights_only=True)
        comparisons[name] = all(torch.equal(v, models['baseline'].__dict__[name].state_dict()[k]) for k, v in expected.items())
    targets = torch.load(BASELINE/'checkpoints'/'targets_5200.pt', map_location='cpu', weights_only=True)
    for name in ('actor_target', 'critic_target'):
        comparisons[name] = all(torch.equal(v, models['baseline'].__dict__[name].state_dict()[k]) for k, v in targets[name].items())
    assert all(comparisons.values()), comparisons
    torch.testing.assert_close(random_states['baseline'], random_states['step_guard'], atol=0, rtol=0)
    write(root/'training_proof.json', {'baseline_networks_exact_A5200': comparisons,
        'same_sampling_and_target_noise': True, 'same_final_torch_rng': True,
        'critic_updates_each': 200, 'actor_proposals_each': 100, 'accepted_actor_updates': accepted_counts,
        'task_contract_unchanged': True, 'physical_online_training': False})


def evaluation(root):
    env = env_for(load(6000), training_course=False, pool_split='validation')
    collector = LowerCollector(env)
    results = {}
    existing = json.loads(Path('analysis_outputs/full_scratch_ego20x10_d3_critic_lead_200/validation.json').read_text())
    results['initial'] = copy.deepcopy(existing['A5000'])
    for group in results['initial'].values():
        for episode in group:
            episode['reused_identical_actor_from'] = 'D3 A5000, same original contract'
    try:
        for label in ('baseline', 'step_guard'):
            for count in (1, 10, 50, 100):
                step = 5000+2*count-1
                agent = checkpoint(root/label, step)
                key = f'{label}_actor{count}'
                results[key] = {}
                tasks = ('L0', 'original16m') if count == 100 else ('L0',)
                for task in tasks:
                    episodes = []
                    for index in range(20):
                        obs, _ = env.reset(seed=index)
                        if task == 'L0':
                            path = env.pool.path_for('validation', index, 0)
                            remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)[::-1])[::-1], 0.]
                            node = int(np.argmin(np.abs(remaining-.8)))
                            obs = relocate_reset(env, path[node])
                        episode = eval_episode(env, agent.actor, obs, collector)
                        upper, lower = episode.pop('upper'), episode.pop('lower')
                        (root/'validation_traces').mkdir(exist_ok=True)
                        np.savez_compressed(root/'validation_traces'/f'{key}_{task}_map{index:02d}.npz',
                            upper_action=np.array([t['action'] for t in upper]),
                            lower_position=np.array([t['position'] for t in lower]),
                            lower_velocity=np.array([t['velocity'] for t in lower]),
                            lower_distance=np.array([t['distance'] for t in lower]))
                        episode['actor_sha256'] = fingerprint(agent.actor)
                        episodes.append(episode)
                        print(f'[Guard validation] {key} {task} map={index} {episode["done_reason"]}', flush=True)
                    results[key][task] = episodes
                    write(root/'validation.json', results)
    finally:
        collector.close()
        env.close()


def finish(root):
    guarded = json.loads((root/'protected_inputs.json').read_text())
    changed = [p for p, digest in guarded.items() if sha(Path(p)) != digest]
    write(root/'protection_result.json', {'unchanged': not changed, 'changed': changed,
                                         'checked_files': len(guarded), 'test_episodes': 0})
    if changed:
        raise RuntimeError(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=('all', 'train', 'evaluation'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    # Original 16m A reproduction used two CPU threads. Match its reduction
    # order so the no-guard control can be verified bit for bit.
    torch.set_num_threads(2)
    if args.part in ('all', 'train'):
        prepare(args.output)
        train(args.output)
    if args.part in ('all', 'evaluation'):
        evaluation(args.output)
    finish(args.output)


if __name__ == '__main__':
    main()
