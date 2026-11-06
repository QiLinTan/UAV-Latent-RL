"""Combine completed unfreeze audits and export scientific diagnostic plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.audit_full_ft_v1_critic_actor import dump
from scripts.reproduce_full_ft_v1_unfreeze import ROOT, STEPS


def main():
    summaries = {seed: json.loads((ROOT / f'seed{seed}' / 'summary.json').read_text()) for seed in (43, 45)}
    result = {'protocol': {'stages': [0, 3], 'diagnostic_map_seeds': [80, 88],
        'checkpoint_steps': list(STEPS), 'actor_updates': [0, 0, 0, 1, 10, 250, 500],
        'critic_4000_updates': 3489, 'source_bank_states_per_seed': 192,
        'gamma_per_upper_transition': .99, 'reward_scale': .1,
        'reward_integration_dt_full': 1 / 120, 'reproduction_threads': 6,
        'offline_and_rollout_threads': 1, 'one_step_continuation': 'same frozen source actor',
        'target_noise_check': 'source actor + normalized N(0,.2), clipped .5, projected; paired four replicas',
        'heldout_environment_seeds_40_79_used': False}, 'seeds': {}}
    for seed, data in summaries.items():
        compact = {'exact_reproduction': data['reproduction']['exact_original_weight_reproduction'],
            'reproduction_trajectory_check': data['reproduction']['trajectory_comparison'],
            'reward_reproduction': data['reproduction']['formal_training_reward_aggregation'],
            'historical_reward': {k: v for k, v in data['historical_reward'].items() if k != 'episodes_detail'},
            'four_k_success_terminal_values': data['offline']['four_k_success_terminal_values'],
            'four_k_replay_scale_verified': data['offline']['four_k_replay_rewards_exactly_original_raw_times_point_one'],
            'calibration': data['calibration'], 'target_noise_check': data['target_noise_check'],
            'first_actor_updates': data['reproduction']['first_actor_updates'], 'frames': {}}
        for step in STEPS:
            frame = data['offline']['frames'][str(step)]
            compact['frames'][str(step)] = {k: v for k, v in frame.items() if k != 'fixed_bank_q'}
            compact['frames'][str(step)]['validation'] = {
                str(stage): {'success': sum(r['success'] for r in data['validation'] if r['step'] == step and r['stage'] == stage),
                             'episodes': 2, 'outcomes': [r['outcome'] for r in data['validation'] if r['step'] == step and r['stage'] == stage]}
                for stage in (0, 3)}
        result['seeds'][str(seed)] = compact
    dump(ROOT / 'summary.json', result)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ticks = np.arange(len(STEPS)); labels = ['0', '3999', '4000', '4001\n1 update', '4019\n10 updates', '4500', '5000']
    for seed, data in summaries.items():
        frames = [data['offline']['frames'][str(s)] for s in STEPS]
        axes[0, 0].plot(ticks, [f['commands']['mean_command_difference_mps'] for f in frames], marker='o', label=f'seed {seed}')
        axes[0, 1].plot(ticks, [f['commands']['updated_mean_goal_closing_command_mps'] for f in frames], marker='o', label=f'seed {seed}')
        axes[1, 0].plot(ticks, [f['td_probe']['q1']['td_rmse'] for f in frames], marker='o', label=f'seed {seed} Q1')
        axes[1, 0].plot(ticks, [f['td_probe']['q2']['td_rmse'] for f in frames], marker='.', linestyle='--', label=f'seed {seed} Q2')
        validation = data['validation']
        for stage in (0, 3):
            axes[1, 1].plot(ticks, [sum(r['success'] for r in validation if r['step'] == s and r['stage'] == stage) / 2 for s in STEPS],
                           marker='o' if stage == 0 else 's', linestyle='-' if stage == 0 else '--', label=f'seed {seed} Stage {stage}')
    for ax, ylabel in zip(axes.ravel(), ('Mean velocity command change (m/s)', 'Mean command toward goal (m/s)', 'TD RMSE, scaled rewards', 'Success fraction, 2 fixed maps / stage')):
        ax.set_xticks(ticks, labels); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.axvline(2.5, color='gray', linestyle=':', alpha=.6)
    fig.suptitle('Full FT v1: exact reproduction of first 5000 transitions')
    fig.savefig(ROOT / 'unfreeze_actor_td_validation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for seed, data in summaries.items():
        branch = json.loads((ROOT / f'seed{seed}' / 'calibration_branches.json').read_text())
        src = [r for r in branch if r['candidate'] == 'source']
        axes[0].scatter([r['discounted_scaled_return'] for r in src], [r['initial_q1_q2_min'][0] for r in src], label=f'seed {seed}')
        pairs = [p for p in data['calibration']['pairwise'] if p['candidate'] == 'q1_best_grid']
        axes[1].scatter([p['actual_return_delta'] for p in pairs], [p['q1_q2_minq_delta'][0] for p in pairs], label=f'seed {seed}')
    axes[0].plot([0, 20], [0, 20], color='gray', linestyle='--'); axes[0].set(xlabel='Actual Full return, common frozen source continuation', ylabel='4000-step Critic Q1', title='Pretrained action value calibration')
    axes[1].axhline(0, color='gray', linewidth=1); axes[1].axvline(0, color='gray', linewidth=1)
    axes[1].set(xlabel='Actual return gain: best-Q action minus source', ylabel='Predicted Q1 gain', title='One forced action, same subsequent policy')
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.savefig(ROOT / 'preunfreeze_q_return_calibration.png', dpi=180); plt.close(fig)
    print(f'Combined summary and two figures: {ROOT}')


if __name__ == '__main__':
    main()
