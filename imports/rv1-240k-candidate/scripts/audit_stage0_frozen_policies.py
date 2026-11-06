"""Evaluate selected frozen actors on the original Stage-0 training map.

This is a fresh policy evaluation on train-map seeds, not a reconstruction of
historical exploratory trajectories. It uses the production CLI to load each
checkpoint, and never performs an optimizer update.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from envs.preprocess import preprocess_state
from scripts import change_td3
from scripts.audit_stage0_feasibility import PENALTIES, make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args
from scripts.run_change_td3 import build_mode_cmd

BASE = 'upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1'


def audit(agent, args, *, seed_count, checkpoint, output):
    env = make_stage0_env(args)
    rows = []
    try:
        for seed in range(seed_count):
            observation, info = env.reset(seed=seed)
            initial_distance = float(info['goal_distance'])
            distances = [initial_distance]
            total_reward = 0.
            term_sum = defaultdict(float)
            for step in range(200):
                action = agent.select_action(preprocess_state(observation.reshape(-1)))
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                distances.append(float(info['goal_distance']))
                terms = env.base_env._last_reward_terms
                for key in ('progress_reward', 'continuous_penalty_total',
                            'terminal_outcome_reward', *PENALTIES):
                    term_sum[key] += float(terms.get(key, 0.))
                if terminated or truncated:
                    break
            assert terminated or truncated
            assert np.isclose(total_reward,
                term_sum['progress_reward']-term_sum['continuous_penalty_total']+
                term_sum['terminal_outcome_reward'], atol=5e-4)
            rows.append(dict(seed=seed, initial_distance_m=initial_distance,
                steps=len(distances)-1, outcome=info['done_reason'],
                final_distance_m=distances[-1], minimum_distance_m=float(min(distances)),
                entered_1p2=bool(min(distances) < 1.2),
                return_raw=total_reward,
                reward_terms={key:float(value) for key,value in term_sum.items()}))
    finally:
        env.close()
    result = dict(checkpoint=checkpoint, train_map=args.map_id, forced_stage=0,
        evaluation_kind='frozen selected actor; no exploration noise; not historical training',
        episodes=rows,
        summary=dict(count=len(rows), outcomes=dict(Counter(r['outcome'] for r in rows)),
            entered_1p2=sum(r['entered_1p2'] for r in rows),
            median_minimum_distance_m=float(np.median([r['minimum_distance_m'] for r in rows])),
            mean_raw_return=float(np.mean([r['return_raw'] for r in rows])),
            mean_reward_terms={key:float(np.mean([r['reward_terms'][key] for r in rows]))
                for key in rows[0]['reward_terms']}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--output', required=True, type=Path)
    cli = parser.parse_args()
    args = preset_args(BASE)
    class AuditCallback(change_td3.EvalCallback):
        def on_step(self, trainer, *, force=False):
            audit(trainer.agent, args, seed_count=cli.seeds,
                  checkpoint=cli.checkpoint, output=cli.output)
    command = build_mode_cmd(BASE)[3:] + [
        '--evaluate_only', 'true', '--load_checkpoint', cli.checkpoint,
        '--log_dir', str(cli.output.parent/f'{cli.output.stem}_loader')]
    with patch.object(change_td3, 'EvalCallback', AuditCallback), patch.object(sys, 'argv', [__file__]+command):
        change_td3.main()


if __name__ == '__main__':
    main()
