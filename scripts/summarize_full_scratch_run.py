"""Read-only audit of the seed43 scratch run; separate raw and restored evals."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = Path('runs/full_scratch_curr_v1_seed43')
CKPT = Path('checkpoints/full_scratch_curr_v1_seed43')
OUT = Path('analysis_outputs/full_scratch_curr_v1_seed43_review_20261013')


def jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    assert np.isfinite(a).all()
    return dict(zip(('min', 'p05', 'median', 'p95', 'max', 'mean'),
                    [*np.quantile(a, [0, .05, .5, .95, 1]).tolist(), float(a.mean())]))


def digest(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def evaluations(path):
    grouped = defaultdict(list)
    for row in jsonl(path):
        assert len(row['episodes']) == 20
        assert sorted(e['environment_seed'] for e in row['episodes']) == list(range(20))
        grouped[row['step']].append(row)
    return [v[0] for v in grouped.values()], {k: v[1:] for k, v in grouped.items() if len(v) > 1}


def training_episodes():
    rows = defaultdict(dict)
    files = []
    for file in sorted(RUN.glob('events*')):
        events = EventAccumulator(str(file), size_guidance={'scalars': 0})
        events.Reload()
        for key in events.Tags()['scalars']:
            if key.startswith('env/') or key.startswith('episode/'):
                for item in events.Scalars(key):
                    rows[item.step][key] = float(item.value)
        values = events.Scalars('episode/return')
        files.append({'file': str(file), 'first_episode_end': values[0].step,
                      'last_episode_end': values[-1].step, 'episodes': len(values)})
    rows = [{'step': step, **row} for step, row in sorted(rows.items()) if 'episode/return' in row]
    return rows, files


def episode_summary(rows):
    if not rows:
        return {'episodes': 0}
    outcomes = Counter(next(k for k in ('success', 'collision', 'timeout', 'height_bound', 'xy_bound', 'attitude_bound')
                            if row.get('env/done_reason/' + k, 0) > .5) for row in rows)
    return {'episodes': len(rows), 'outcomes': dict(outcomes),
            'success_rate': outcomes['success'] / len(rows),
            'collision_rate': outcomes['collision'] / len(rows),
            'guidance_entry_rate': float(np.mean([r['env/training_episode/entered_guidance_1p2'] for r in rows])),
            'final_distance': stats([r['env/goal_distance'] for r in rows])}


def trace_summary(folder, expected):
    rows = []
    for path in sorted(folder.glob('*.json')):
        meta = json.loads(path.read_text())
        if 'environment_seed' not in meta:
            continue
        seed = meta['environment_seed']
        episode = next(e for e in expected['episodes'] if e['environment_seed'] == seed)
        assert meta['done_reason'] == episode['done_reason']
        assert abs(meta['final_goal_distance'] - episode['final_goal_distance']) < 1e-5
        with np.load(path.with_suffix('.npz')) as d:
            clearance = d['lower_clearance']
            pos = d['lower_position']; vel = d['lower_velocity']; cmd = d['lower_command_velocity']
            goal = np.asarray(meta['target_position']) - pos
            dist = np.linalg.norm(goal, axis=1)
            near = dist < 1.2
            direction = goal / np.maximum(dist[:, None], 1e-9)
            tracking = np.linalg.norm(pos - d['lower_reference_position'], axis=1)
            rows.append({'seed': seed, 'success': meta['success'], 'done_reason': meta['done_reason'],
                         'minimum_clearance_m': float(clearance.min()),
                         'tracking_rmse_m': float(np.sqrt(np.mean(tracking**2))),
                         'guidance_entered': bool(near.any()),
                         'near_actual_radial_mps': float(np.sum(vel[near]*direction[near],axis=1).mean()) if near.any() else None,
                         'near_actual_vertical_mps': float(vel[near,2].mean()) if near.any() else None,
                         'near_command_vertical_mps': float(cmd[near,2].mean()) if near.any() else None,
                         'actor_planar_boundary_fraction': float(np.mean(np.linalg.norm(d['upper_policy_action'][:,:2],axis=1)>=.999))})
    assert len(rows) == 20
    successes = [r for r in rows if r['success']]
    return {'episodes': rows, 'successful_episode_minimum_clearance_m': stats([r['minimum_clearance_m'] for r in successes]),
            'successes_below_clearance': {str(th): sum(r['minimum_clearance_m'] < th for r in successes) for th in (.01, .05, .10, .20)},
            'collision_seeds': [r['seed'] for r in rows if r['done_reason'] == 'collision']}


def visited_state_q_diagnostic(critic, discount):
    """Calibration against realized policy returns, not a causal Q-error test."""
    result = []
    for step in (165000, 190000, 240000):
        critic.load_state_dict(torch.load(CKPT / f'model_{step}_critic',
                                          map_location='cpu', weights_only=True))
        folder = (RUN / 'current_stage_traces' / f'step_{step:07d}' if step != 240000
                  else OUT / 'raw240k_stage3/traces')
        obs, actions, returns = [], [], []
        for file in sorted(folder.glob('*.npz')):
            with np.load(file) as data:
                value = 0.
                episode_returns = []
                for reward in data['upper_reward'][::-1] * .1:
                    value = float(reward) + discount * value
                    episode_returns.append(value)
                returns.extend(episode_returns[::-1])
                obs.extend(data['upper_pre_observation'])
                actions.extend(data['upper_action'])
        with torch.no_grad():
            q1, q2 = critic(torch.tensor(np.asarray(obs), dtype=torch.float32),
                            torch.tensor(np.asarray(actions), dtype=torch.float32))
        target = np.asarray(returns)
        row = {'step': step, 'states': len(target), 'discount': discount, 'scaled_reward': .1,
               'on_policy_discounted_return_mean': float(target.mean())}
        for name, value in (('q1', q1), ('q2', q2)):
            value = value.numpy().ravel()
            row[name] = {'mean': float(value.mean()),
                         'p05_p50_p95': np.quantile(value, [.05, .5, .95]).tolist(),
                         'mean_prediction_minus_return': float((value-target).mean()),
                         'rmse_against_return': float(np.sqrt(np.mean((value-target)**2)))}
        result.append(row)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, restored = evaluations(RUN / 'evaluation_metrics.jsonl')
    stage0, stage0_restored = evaluations(RUN / 'stage0_7m/evaluation_metrics.jsonl')
    episodes, files = training_episodes()
    windows = []
    for lo, hi in ((0,2000),(2000,5000),(5000,10000),(10000,20000),(20000,24000),(24000,32000),
                   (32000,60000),(60000,90000),(90000,130000),(130000,160000),(160000,190000),(190000,220000),(220000,240000)):
        selected = [r for r in episodes if lo < r['step'] <= hi]
        windows.append({'lo': lo, 'hi': hi, **episode_summary(selected),
                        'band': episode_summary([r for r in selected if r['env/start_curriculum_is_band']>.5]),
                        'original': episode_summary([r for r in selected if r['env/start_curriculum_is_original']>.5])})
    milestones = []
    last = None
    resets = list(jsonl(RUN / 'p0_audit/resets.jsonl'))
    for row in resets:
        key = (row['level'], row['stage'])
        if key != last:
            milestones.append(row); last = key
    buckets = defaultdict(lambda: defaultdict(list))
    first_conservative = None
    nonfinite = []; frozen = 0; update_count = 0; actor_count = 0
    keys = ('critic_loss','critic_grad_norm','actor_grad_norm','q1_mean','q2_mean','q1_min','q1_max','q2_min','q2_max',
            'td1_abs_mean','td2_abs_mean','q_gap_abs_mean','actor_sat_pct','actor_planar_boundary_pct','actor_policy_anchor_loss')
    success_sizes = {}
    previous = None
    for row in jsonl(RUN / 'p0_audit/updates.jsonl'):
        if previous is not None:
            assert row['step'] == previous + 1
        previous = row['step']; update_count += 1; actor_count += int(row['actor_updated'])
        if row['conservative_phase'] and first_conservative is None:
            first_conservative = row['step']
        frozen += int(row['actor_frozen'])
        for key, value in row.items():
            if value is not None and not np.isfinite(value):
                nonfinite.append([row['step'],key])
        bucket = (row['step']-1)//5000*5000
        for key in keys:
            if row[key] is not None and (key != 'actor_grad_norm' or row['actor_updated']):
                buckets[bucket][key].append(row[key])
        if row['step'] in (5000,10000,20000,24000,30000,60000,100000,130000,240000):
            success_sizes[row['step']] = {k:v for k,v in row.items() if k.startswith('success_replay/level_') and k.endswith('_size')}
    stability = [{'lo':k,'hi':k+5000,**{name:stats(vals) for name,vals in v.items()}} for k,v in sorted(buckets.items())]
    policy = {}
    for row in jsonl(RUN/'p0_audit/policy_diagnostics.jsonl'):
        policy.setdefault(row['step'], row)
    snapshot = torch.load(CKPT/'training_state.pt', map_location='cpu', weights_only=False)
    hashes = {name:{kind:digest(torch.load(CKPT/f'{name}_{kind}',map_location='cpu',weights_only=True))
                   for kind in ('actor','critic')} for name in
              ('model_initial','model_5000','model_10000','model_165000','model_190000','model_240000',
               'model_final','model_best','model_peak_validation','model_raw_final')}
    assert hashes['model_final']==hashes['model_best']==hashes['model_165000']
    assert hashes['model_raw_final']==hashes['model_240000']
    assert hashes['model_peak_validation']==hashes['model_190000']
    assert snapshot['steps']==240000 and snapshot['agent']['total_it']==update_count==235001
    assert hashes['model_raw_final']=={k:digest(snapshot['agent'][k].state_dict()) for k in ('actor','critic')}
    safety = {}
    for row in raw:
        step = row['step']
        if step < 130000:
            continue
        folder = RUN/'current_stage_traces'/f'step_{step:07d}'
        if step == 240000:
            folder = OUT/'raw240k_stage3/traces'
            repeated = list(jsonl(OUT/'raw240k_stage3/evaluation_metrics.jsonl'))[-1]
            for a,b in zip(row['episodes'],repeated['episodes']):
                assert a['environment_seed']==b['environment_seed'] and a['done_reason']==b['done_reason']
                assert abs(a['final_goal_distance']-b['final_goal_distance'])<1e-5
        safety[step] = trace_summary(folder,row)
    result = {'cumulative_steps':snapshot['steps'],'checkpoint_hashes':hashes,
              'critic_updates':update_count,'actor_updates':actor_count,'frozen_updates':frozen,
              'nonfinite_updates':nonfinite,'first_conservative_update':first_conservative,
              'replay_size':snapshot['replay']['size'],'success_replay_final':{k:v['size'] for k,v in snapshot['success']['strata'].items()},
              'course_milestones':milestones,'event_files':files,'training_windows':windows,
              'first_post_warmup_training_success':next(r['step'] for r in episodes if r['step']>5000 and r['env/success']>.5),
              'success_replay_at_checkpoints':success_sizes,'raw_evaluations':raw,'restored_evaluations':restored,
              'stage0_evaluations':stage0,'stage0_restored_evaluations':stage0_restored,
              'training_stability':stability,'fixed_policy_probes':list(policy.values()),'stage3_safety':safety,
              'selection':json.loads((CKPT/'model_final_selection.json').read_text())}
    result['visited_state_q_diagnostic'] = visited_state_q_diagnostic(
        snapshot['agent']['critic'], snapshot['agent']['discount'])
    (OUT/'visited_state_q_diagnostic.json').write_text(
        json.dumps(result['visited_state_q_diagnostic'], indent=2) + '\n')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result)
    print(json.dumps({k:result[k] for k in ('cumulative_steps','critic_updates','actor_updates','first_conservative_update',
                                          'first_post_warmup_training_success','training_windows','success_replay_at_checkpoints')},indent=2))
    print('SAFETY',json.dumps({k:{x:y for x,y in v.items() if x!='episodes'} for k,v in safety.items()},indent=2))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(3,2,figsize=(13,11))
    raw=data['raw_evaluations']; stage0=data['stage0_evaluations']; x=np.array([r['step'] for r in raw])/1000
    ax[0,0].plot(x,[r['summary']['eval/success_rate'] for r in raw],'-o',ms=3,label='Current stage, raw')
    ax[0,0].plot(x,[r['summary']['eval/success_rate'] for r in stage0],label='Stage 0 / 7m, raw',alpha=.8)
    ax[0,0].set_title('Fixed validation success (20 maps)');ax[0,0].set_ylim(-.05,1.05);ax[0,0].legend()
    ax[0,1].plot(x,[r['summary']['eval/collision_rate'] for r in raw],label='Collision')
    ax[0,1].plot(x,[r['summary']['eval/done_reason_timeout_rate'] for r in raw],label='Timeout')
    ax[0,1].set_title('Current-stage failure rates');ax[0,1].legend()
    rows=data['course_milestones'];mx=[r['step']/1000 for r in rows]+[240]
    ax[1,0].step(mx,[r['level'] for r in rows]+[4],where='post',label='Start level')
    ax[1,0].step(mx,[r['stage'] for r in rows]+[3],where='post',label='Obstacle stage')
    ax[1,0].set_title('Actual curriculum progression');ax[1,0].legend()
    safety=data['stage3_safety'];sx=np.array(list(safety))/1000
    for key in ('min','median'):
        ax[1,1].plot(sx,[v['successful_episode_minimum_clearance_m'][key] for v in safety.values()],label=key)
    ax[1,1].axhline(.05,ls='--',color='gray',label='5 cm');ax[1,1].set_title('Stage 3: successful-episode minimum clearance (m)');ax[1,1].legend()
    rows=data['training_stability'];tx=[r['hi']/1000 for r in rows]
    ax[2,0].plot(tx,[r['td1_abs_mean']['median'] for r in rows],label='TD1 abs, median')
    ax[2,0].plot(tx,[r['td2_abs_mean']['median'] for r in rows],label='TD2 abs, median')
    ax[2,0].set_title('Training TD errors (5k windows)');ax[2,0].legend()
    probes=data['fixed_policy_probes'];px=[r['step']/1000 for r in probes]
    ax[2,1].plot(px,[r['actor_vertical_command_mps'] for r in probes],label='Vertical command (m/s)')
    ax[2,1].plot(px,[r['adjacent_policy_rms_change'] or 0 for r in probes],label='Adjacent policy RMS change')
    ax[2,1].set_title('Fixed P0-state probe (distribution limited)');ax[2,1].legend()
    for a in ax.flat:
        a.set_xlabel('Cumulative upper transitions (k)');a.grid(alpha=.25)
        a.axvline(128.958,color='gray',lw=.7,ls=':')
    fig.suptitle('Full scratch seed43 | raw checkpoints | selected final = 165k | no held-out claim')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'learning_audit.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
