"""Assert actual P0 state and produce evidence, without touching the learner."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.forest.start_curriculum import StartDistanceCurriculum
from envs.forest.ego_curriculum_env import BANDS
from envs.map_atlas.forest_pool import ForestPool, COUNTS
from trainers.full_scratch import fingerprint


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


def metrics(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p05', 'p50', 'p95', 'max'), np.quantile(values, [0, .05, .5, .95, 1]).tolist())) if values.size else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='full_scratch_ego20x10_curr_v1_seed43')
    parser.add_argument('--output', default='analysis_outputs/full_scratch_ego20x10_curr_v1_p0/report.json')
    args = parser.parse_args()
    root = Path('checkpoints') / args.run
    logs = Path('runs') / args.run
    audit = logs / 'p0_audit'
    pool = ForestPool()
    initial = json.loads((root / 'initialization.json').read_text())
    two = torch.load(root / 'training_state_step2000.pt', map_location='cpu', weights_only=False)
    six = torch.load(root / 'training_state_step6000.pt', map_location='cpu', weights_only=False)
    resume = json.loads((root / 'resume_at_2000.json').read_text())
    transitions = lines(audit / 'transitions.jsonl')
    episodes = lines(audit / 'episodes.jsonl')
    resets = lines(audit / 'resets.jsonl')
    updates = lines(audit / 'updates.jsonl')
    assert two['steps'] == two['replay']['size'] == 2000 and two['agent']['total_it'] == 0
    assert initial['pretrained_source'] is None and initial['actor_optimizer_states'] == initial['critic_optimizer_states'] == 0
    assert fingerprint(two['agent']['actor']) == initial['actor'] and fingerprint(two['agent']['critic']) == initial['critic']
    assert six['steps'] == six['replay']['size'] == len(transitions) == 6000
    assert [r['step'] for r in transitions] == list(range(1, 6001))
    assert [r['step'] for r in updates] == list(range(5000, 6001))
    assert len(updates) == six['agent']['total_it'] == 1001
    assert sum(r['actor_updated'] for r in updates) == 500
    assert all(r['actor_frozen'] == r['conservative_phase'] == 0 for r in updates)
    assert all(v is None or np.isfinite(v) for row in updates for v in row.values())
    assert len(six['agent']['actor_optimizer'].state) > 0 and len(six['agent']['critic_optimizer'].state) > 0
    assert resume['pool_sampler'] == json.loads(json.dumps(two['pool_sampler']))
    assert resume['replay_size'] == resume['replay_ptr'] == 2000
    assert resume['actor'] == initial['actor'] and resume['critic'] == initial['critic']
    for key, value in two['replay']['arrays'].items():
        np.testing.assert_array_equal(value, six['replay']['arrays'][key][:2000])
    train_ids = {e['record']['map_id'] for e in pool.entries('train')}
    assert all(r['map_split'] == 'train' and r['map_id'] in train_ids and r['active_tree_count'] == COUNTS[r['density_stage']] for r in transitions)
    np.testing.assert_allclose(np.asarray([r['command_velocity'] for r in transitions]),
                               six['replay']['arrays']['action'] * np.array([.8, .8, .3]), atol=1e-7)
    assert six['pool_sampler']['fingerprint'] == pool.fingerprint
    assert six['pool_sampler']['draws'] == len(resets)
    # Reconstruct gates from actual completed production episodes only.
    start = StartDistanceCurriculum(bands=BANDS)
    density = ObstacleCurriculum(route_counts=COUNTS)
    for row in episodes:
        assert row['start_level'] == start.level and row['density_stage'] == density.stage
        complete = row['start_level'] >= len(BANDS)
        start.record(source=row['start_source'], success=row['done_reason'] == 'success', collision=row['done_reason'] == 'collision')
        density.record(episode_stage=row['density_stage'], start_level=row['start_level'], start_complete=complete and row['start_source'] == 'original',
                       primary_difficulty=row['primary_difficulty'], success=row['done_reason'] == 'success', collision=row['done_reason'] == 'collision')
    for expected, actual in ((start, six['start']), (density, six['obstacle'])):
        for key in ('successes', 'collisions'):
            assert list(getattr(expected, key)) == list(actual[key])
    assert start.level == six['start']['level'] and density.stage == six['obstacle']['stage']
    for count, stratum in six['success']['strata'].items():
        expected_size = min(stratum['max_size'], sum(e['length'] for e in episodes if e['done_reason'] == 'success' and e['active_tree_count'] == count))
        assert stratum['size'] == expected_size
    protection = Path(args.output).parent / 'protected_checkpoint_hashes.json'
    protected = json.loads(protection.read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in protected.items())
    networks = {}
    for name in ('actor', 'critic', 'actor_target', 'critic_target'):
        networks[name] = dict(initial=fingerprint(two['agent'][name]), final=fingerprint(six['agent'][name]))
        assert networks[name]['initial'] != networks[name]['final']
    assert networks['actor']['final'] != networks['actor_target']['final']
    assert networks['critic']['final'] != networks['critic_target']['final']
    report = dict(hard_contract='PASS', pool_fingerprint=pool.fingerprint, protected_artifact_count=len(protected),
                  initialization=initial, networks=networks, replay_size=6000,
                  success_replay_sizes={k: v['size'] for k, v in six['success']['strata'].items()},
                  critic_updates=1001, actor_updates=500, actor_frozen=False,
                  actor_adam_states=len(six['agent']['actor_optimizer'].state), critic_adam_states=len(six['agent']['critic_optimizer'].state),
                  sampler_draw_count=six['pool_sampler']['draws'], unique_train_maps=len({r['map_id'] for r in resets}),
                  start_level=start.level, start_gate_episodes=start.level_episodes,
                  density_stage=density.stage, density_gate_episodes=density.stage_episodes,
                  interrupted_episode_steps_at_2k=two['interrupted_episode_steps'],
                  interrupted_episode_steps_at_6k=six['interrupted_episode_steps'],
                  resume=resume, policy_probes=lines(audit / 'policy_diagnostics.jsonl'), phases={}, evaluations=[])
    for label, begin, end in (('2k', 0, 2000), ('2k_to_5k', 2000, 4999), ('online_5k_to_6k', 4999, 6000), ('all', 0, 6000)):
        eps = [e for e in episodes if begin < e['step'] <= end]
        report['phases'][label] = dict(complete_episodes=len(eps), outcomes=dict(Counter(e['done_reason'] for e in eps)),
                                      sources=dict(Counter(e['start_source'] for e in eps)),
                                      band_outcomes=dict(Counter(e['done_reason'] for e in eps if e['start_source'] == 'band')),
                                      guidance_entered=sum(e['minimum_goal_distance'] < 1.2 for e in eps),
                                      final_distance=metrics([e['goal_distance'] for e in eps]),
                                      minimum_clearance=metrics([e['episode_minimum_tree_clearance'] for e in eps]))
    report['update_metrics'] = {key: metrics([r[key] for r in updates]) for key in ('critic_loss', 'critic_grad_norm', 'actor_grad_norm',
                                 'q1_mean', 'q2_mean', 'q1_min', 'q2_min', 'q1_max', 'q2_max', 'td1_abs_mean', 'td2_abs_mean', 'td1_abs_p95', 'td2_abs_p95')}
    report['last_update'] = updates[-1]
    for path in sorted((logs / 'validation').glob('*/*/density*_metrics.json')):
        data = json.loads(path.read_text())
        assert all(e['map_split'] == 'validation' for e in data['episodes'])
        report['evaluations'].append({k: data[k] for k in ('step', 'source_step', 'context', 'actor_sha256', 'density_stage', 'summary')})
    report['selection'] = json.loads((root / 'model_final_selection.json').read_text()) if (root / 'model_final_selection.json').exists() else None
    report['learning_release'] = 'HOLD: post-warmup bound failures and positive-z saturation; audit before extending budget'
    corrected = audit / 'policy_diagnostics_world_corrected.json'
    if corrected.exists():
        report['corrected_policy_probes'] = json.loads(corrected.read_text())
    report['implementation_files_sha256'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in ('envs/map_atlas/forest_pool.py', 'envs/forest/ego_curriculum_env.py',
                     'envs/forest/start_curriculum.py', 'envs/forest/start_curriculum_env.py',
                     'trainers/full_scratch.py', 'trainers/td3_trainer.py', 'trainers/callbacks/ego_pool_eval.py',
                     'algos/td3/td3_upper_recurrent.py', 'scripts/change_td3.py', 'scripts/run_change_td3.py')}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('hard_contract', 'replay_size', 'success_replay_sizes', 'critic_updates', 'actor_updates', 'unique_train_maps', 'start_level', 'density_stage', 'phases')}, indent=2))


if __name__ == '__main__':
    main()
