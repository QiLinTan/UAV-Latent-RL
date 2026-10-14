"""Summarize frozen-model multistep probes and logged training stability."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = 'upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1'
OUT = Path('analysis_outputs/value_history_20261008')


def statistics(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return dict(count=len(a), nonfinite=int(len(a)-len(finite)),
        median=float(np.median(finite)) if len(finite) else None,
        p95=float(np.percentile(finite,95)) if len(finite) else None,
        maximum=float(np.max(finite)) if len(finite) else None,
        mean=float(np.mean(finite)) if len(finite) else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in (42,43,44,45):
        name = BASE if seed==42 else f'{BASE}_seed{seed}'
        series = defaultdict(list)
        for file in sorted(Path(f'runs/{name}').glob('events.out.tfevents.*')):
            acc = EventAccumulator(str(file), size_guidance={'scalars':0})
            acc.Reload()
            for tag in ('critic_loss','actor_loss','critic_grad_norm','actor_grad_norm',
                        'q1_mean','q2_mean','q_target_mean','q_gap_abs_mean',
                        'actor_planar_boundary_pct','actor_lr','critic_lr','conservative_phase'):
                if tag in acc.Tags()['scalars']:
                    series[tag].extend(dict(step=r.step, value=r.value) for r in acc.Scalars(tag))
        stats = {}
        for tag, rows in series.items():
            stats[tag] = statistics([r['value'] for r in rows])
            stats[tag]['windows'] = {str(start):statistics([r['value'] for r in rows if start<=r['step']<start+20000])
                                     for start in range(0,120000,20000)}
            if tag.endswith('grad_norm'):
                positive = [r['value'] for r in rows if r['value']>0]
                stats[tag]['fraction_above_clip_1_given_positive'] = float(np.mean(np.array(positive)>1)) if positive else None
        (OUT/f'training_series_seed{seed}.json').write_text(json.dumps(series, indent=2)+'\n')
        folder = Path(f'analysis_outputs/value_history_seed{seed}_20261008')
        retry = Path(str(folder)+'_retry')
        if (retry/'summary.json').exists():
            folder = retry
        if not (folder/'summary.json').exists():
            results.append(dict(seed=seed, training=stats, pending_frozen_audit=True))
            continue
        audit = json.loads((folder/'summary.json').read_text())
        rows = json.loads((folder/'branches.json').read_text())
        behaviors = defaultdict(list)
        groups = defaultdict(list)
        for row in rows:
            label = f"{row['behavior']}_{row['forced_steps']}" if row['behavior']=='planned' else row['behavior']
            behaviors[label].append(row)
            groups[(row['seed'],row['start'])].append(row)
        behavior_stats = {}
        for label, subset in behaviors.items():
            behavior_stats[label] = dict(count=len(subset), outcomes=dict(Counter(r['outcome'] for r in subset)),
                mean_return=float(np.mean([r['return_actual'] for r in subset])),
                mean_final_distance=float(np.mean([r['final_distance'] for r in subset])))
            if label != 'planned_full':
                behavior_stats[label]['bootstrap_q1_mae'] = float(np.mean([
                    abs(r['prefix_plus_actor_bootstrap'][0]-r['return_actual']) for r in subset]))
        # Forced full trajectories have no independent bootstrap, so exclude
        # them from the model-ranking test (their estimate equals measured G).
        agree = disagree = 0
        for subset in groups.values():
            subset = [r for r in subset if r['behavior']!='planned_full' and len(r['rewards'])>r['forced_steps']]
            for i, a in enumerate(subset):
                for b in subset[i+1:]:
                    dp = a['prefix_plus_actor_bootstrap'][0]-b['prefix_plus_actor_bootstrap'][0]
                    dg = a['return_actual']-b['return_actual']
                    if abs(dp)>.01 and abs(dg)>.01:
                        agree += int(dp*dg>0)
                        disagree += int(dp*dg<0)
        actor_rows = [r for r in rows if r['behavior']=='actor']
        nstep = {}
        for h in (0,1,5,15,45):
            predicted = [r['prefix_plus_actor_bootstrap'][0] if h==0 else r['actor_nstep_estimates'][str(min(h,len(r['rewards'])))][0] for r in actor_rows]
            actual = [r['return_actual'] for r in actor_rows]
            nstep[str(h)] = dict(mean_error=float(np.mean(np.array(predicted)-actual)),
                                mae=float(np.mean(np.abs(np.array(predicted)-actual))))
        results.append(dict(seed=seed, training=stats, audit=audit, behaviors=behavior_stats,
                            prefix_ranking=dict(agree=agree, disagree=disagree), actor_nstep=nstep))
    (OUT/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    for r in results:
        print('SEED',r['seed'],'pending',r.get('pending_frozen_audit',False))
        print('training', {k:{a:b for a,b in v.items() if a!='windows'} for k,v in r['training'].items()})
        print('behaviors',r.get('behaviors'))
        print('nstep',r.get('actor_nstep'),'ranking',r.get('prefix_ranking'))


if __name__=='__main__':
    main()
