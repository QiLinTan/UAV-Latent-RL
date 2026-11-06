"""Read-only completion and regression audit of the two Full FT v1 pilot runs."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent

ROOT = Path('analysis_outputs/full_ft_v1_first_round_check_20261012')
SOURCE = Path('checkpoints/startobscurr_v1_selected_20261010')
TASKS = ('full_stage3_validation', 'full_stage0_validation', 'k1_stage3_validation')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def optimizer_steps(path):
    data = load(path)
    return sorted({int(state['step']) for state in data['state'].values()})


def source_policy_drift(seed):
    agent = TD3UpperRecurrent(
        917, 3, 1.0, base_state_dim=117, history_length=8,
        history_frame_dim=100, hidden_dim=64, project_planar_action=True,
        device=torch.device('cpu'),
    )
    agent.load_actor_only(str(SOURCE / f'seed{seed}' / 'model_final'))
    steps, differences = [], []
    for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json')):
        meta = json.loads(path.read_text())
        first_step = meta['global_upper_step'] - meta['upper_steps'] + 1
        if first_step > 5500:
            break
        with np.load(path.with_suffix('.npz')) as trace:
            obs = trace['upper_pre_observation']
            actions = trace['upper_policy_action']
        with torch.no_grad():
            expected = agent.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        steps.extend(range(first_step, first_step + len(actions)))
        differences.extend(np.max(np.abs(expected - actions), axis=1))
    steps, differences = np.asarray(steps), np.asarray(differences)
    frozen_difference = float(differences[steps <= 4000].max())
    assert frozen_difference < 1e-5
    return {
        'frozen_phase_max_action_difference_vs_source': frozen_difference,
        'first_policy_action_change_upper_step': int(steps[differences > 1e-5][0]),
        'after_unfreeze_action_difference_bands': [
            {'first_step': lo, 'last_step': hi,
             'mean_max_axis_difference': float(differences[(steps >= lo) & (steps <= hi)].mean())}
            for lo, hi in ((4001, 4250), (4251, 4500), (4501, 5000))
        ],
        'difference_definition': 'max absolute normalized action-axis difference on the same logged Full pre-observation',
    }


def audit(seed, manifest):
    checkpoint = Path(f'checkpoints/full_ft_v1_seed{seed}')
    actor_file = SOURCE / f'seed{seed}' / 'model_final_actor'
    digest = hashlib.sha256(actor_file.read_bytes()).hexdigest()
    assert digest == manifest['seeds'][str(seed)]['files']['model_final_actor']['sha256']
    source_actor = load(actor_file)
    matches = {}
    for name in ('model_initial', 'model_peak_validation', 'model_5000', 'model_40000', 'model_raw_final', 'model_final'):
        actor = load(checkpoint / f'{name}_actor')
        matches[name] = all(torch.equal(actor[key], source_actor[key]) for key in source_actor)
    assert matches['model_initial'] and matches['model_final']
    raw = load(checkpoint / 'model_raw_final_actor')
    last = load(checkpoint / 'model_40000_actor')
    assert all(torch.equal(raw[key], last[key]) for key in raw)
    assert optimizer_steps(checkpoint / 'model_initial_actor_optimizer') == []
    assert optimizer_steps(checkpoint / 'model_initial_critic_optimizer') == []
    result = {
        'completed_upper_steps': 40000, 'source_actor_manifest_sha256_verified': True,
        'actor_matches_source': matches, 'raw_final_equals_model_40000_actor': True,
        'initial_optimizers_empty': True,
        'raw_actor_optimizer_steps': optimizer_steps(checkpoint / 'model_raw_final_actor_optimizer'),
        'raw_critic_optimizer_steps': optimizer_steps(checkpoint / 'model_raw_final_critic_optimizer'),
        'model_final_selection': json.loads((checkpoint / 'model_final_selection.json').read_text()),
        'validation': {}, 'source_policy_drift': source_policy_drift(seed),
    }
    for task in TASKS:
        path = Path(f'runs/full_ft_v1_seed{seed}') / task / 'evaluation_metrics.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert [row['step'] for row in records] == [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 40000]
        rows = []
        for i, record in enumerate(records):
            assert [ep['environment_seed'] for ep in record['episodes']] == list(range(80, 90))
            data = record['summary']
            rows.append({
                'upper_step': record['step'],
                'phase': 'restored_selected' if i == 9 else 'initial' if i == 0 else 'raw_periodic',
                'success_count': sum(bool(ep['success']) for ep in record['episodes']),
                'collision_count': sum(bool(ep['collision']) for ep in record['episodes']),
                'success_rate': data['eval/success_rate'],
                'collision_rate': data['eval/collision_rate'],
                'mean_final_goal_distance_m': data['eval/final_goal_distance'],
                'outcomes': dict(Counter(ep['done_reason'] for ep in record['episodes'])),
            })
        result['validation'][task] = rows
    traces = [json.loads(path.read_text()) for path in sorted(Path(f'analysis_outputs/full_ft_v1_seed{seed}/training_traces').glob('episode_*/*.json'))]
    assert sum(row['upper_steps'] for row in traces) == 40000
    assert all(row['mode'] == 'full' and row['upper_rate_hz'] == 15 and row['lower_rate_hz'] == 120 for row in traces)
    assert all(row['initial_position'] == [-3.5, 0.0, 1.0] and row['target_position'] == [3.5, 0.0, 1.0] for row in traces)
    completed = [row for row in traces if row['done_reason'] != 'training_budget_end']
    for i in range(0, len(completed) - 9, 10):
        assert [row['curriculum_stage'] for row in completed[i:i + 10]].count(0) == 3
    result['training'] = {
        'completed_episodes': len(completed), 'partial_episodes': len(traces) - len(completed),
        'recorded_upper_transitions': sum(row['upper_steps'] for row in traces),
        'recorded_lower_steps': sum(row['lower_steps'] for row in traces),
        'completed_stage_counts': dict(Counter(row['curriculum_stage'] for row in completed)),
        'completed_outcomes': dict(Counter(row['done_reason'] for row in completed)),
        'episode_end_step_bands': [
            {'exclusive_lower_step': low, 'inclusive_upper_step': high,
             'episodes': sum(low < row['global_upper_step'] <= high for row in completed),
             'outcomes': dict(Counter(row['done_reason'] for row in completed if low < row['global_upper_step'] <= high))}
            for low, high in ((0, 4000), (4000, 10000), (10000, 40000))
        ],
    }
    return result


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    summary = {
        'date': '2026-10-12', 'validation_environment_seeds': list(range(80, 90)),
        'heldout_environment_seeds_40_79_evaluated': Path('analysis_outputs/full_ft_v1_heldout_20261010').exists(),
        'seeds': {str(seed): audit(seed, manifest) for seed in (43, 45)},
        'trace_limitation': 'Final restored evaluation uses the same step_0040000 trace directory as raw final validation; JSONL retains both evaluations, but raw-final validation traces were overwritten.',
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, seed in zip(axes, (43, 45)):
        for task, label in zip(TASKS, ('Full Stage 3', 'Full Stage 0', 'K1 Stage 3')):
            rows = summary['seeds'][str(seed)]['validation'][task][:-1]
            ax.plot([row['upper_step'] / 1000 for row in rows], [row['success_rate'] * 100 for row in rows], marker='o', label=label)
        ax.axvspan(0, 4, color='gray', alpha=.12, label='Actor frozen')
        ax.set(title=f'Training seed {seed}', xlabel='Upper transitions (thousands)', ylim=(-3, 105), xticks=range(0, 41, 5))
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation success (%)')
    fig.suptitle('Full FT v1: raw policy validation; restored final excluded')
    fig.tight_layout(); fig.savefig(ROOT / 'validation_success.png', dpi=180); plt.close(fig)
    print(ROOT / 'summary.json')


if __name__ == '__main__':
    main()
