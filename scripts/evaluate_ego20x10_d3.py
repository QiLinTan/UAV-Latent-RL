"""D3 finite-horizon value probes and paired validation. Never accesses test split."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.audit_ego20x10_value import (
    DEFAULT as BASELINE, LOGS, ZGRID, load, learner, env_for, relocate_reset, reconstruct_state,
    preprocess_state, rows, world_goal_delta, describe, fingerprint, write,
)
from scripts.run_ego20x10_d3 import ROOT, checkpoint


def agents(root):
    source=load(5000)
    result={'A5000':learner(source),'A6000':learner(load(6000))}
    middle=learner(load(5000))
    for name in ('actor','critic'):
        getattr(middle,name).load_state_dict(torch.load(BASELINE/'checkpoints'/f'model_raw_5200_{name}',map_location='cpu',weights_only=True))
    result['A5200']=middle
    for step in (5200,5325,5600,6000):result[f'B{step}']=checkpoint(root/'B',step)
    return result


def finite_rollout(env,observation,first,actor,horizon=90):
    total=0.;limits={};samples=[];info={};terminal=False
    for tick in range(horizon):
        if tick==0:action=first
        else:
            with torch.no_grad():action=actor(torch.as_tensor(preprocess_state(observation)).reshape(1,-1))[0].numpy()
        observation,reward,terminated,truncated,info=env.step(action.reshape(1,3))
        total+=.1*reward*.99**tick
        terminal=bool(terminated or truncated)
        samples.append(dict(action=action.tolist(),position=env.pos[0].tolist(),velocity=env.vel[0].tolist(),
                            reference_tracking_error=info['reference_tracking_error_mean'],raw_reward=float(reward)))
        if tick+1 in (30,90) or terminal:limits[str(tick+1)]=dict(discounted_scaled_return=total,terminal=terminal)
        if terminal:break
    for limit in (30,90):
        if str(limit) not in limits:limits[str(limit)]=dict(discounted_scaled_return=total,terminal=terminal)
    return dict(returns=limits,actual_upper_steps=tick+1,done_reason=info['done_reason'] if terminal else 'diagnostic_horizon',
                final_distance=float(info['goal_distance']),samples=samples)


def rank_pairs(q,returns):
    q=np.asarray(q);returns=np.asarray(returns)
    concordant=discordant=ties=0
    for i in range(5):
        for j in range(i+1,5):
            if abs(q[j]-q[i])<1e-7 or abs(returns[j]-returns[i])<1e-6:ties+=1
            elif (q[j]-q[i])*(returns[j]-returns[i])>0:concordant+=1
            else:discordant+=1
    rho=float(spearmanr(q,returns).statistic) if np.ptp(q)>0 and np.ptp(returns)>0 else None
    return dict(concordant=concordant,discordant=discordant,ties=ties,spearman=rho,
                q_top_z=float(ZGRID[np.argmax(q)]),return_top_z=float(ZGRID[np.argmax(returns)]),
                top_action_agreement=bool(abs(q[np.argmax(returns)]-q.max())<1e-7))


def ranking_summary(records):
    raw=[];summary={}
    for record in records:
        for label,values in record['scans'].items():
            for suffix in ('common','own'):
                for horizon in (30,90):
                    returns=[s['physical'][suffix]['returns'][str(horizon)]['discounted_scaled_return'] for s in values]
                    for channel in ('q1','q2','q_min'):
                        raw.append(dict(checkpoint=label,suffix=suffix,horizon=horizon,critic=channel,
                                        index=record['index'],groups=record['groups'],**rank_pairs([s[channel] for s in values],returns)))
    for label in sorted({r['checkpoint'] for r in raw}):
        summary[label]={}
        for group in ('all','near','far','above','below'):
            summary[label][group]={}
            for suffix in ('common','own'):
                summary[label][group][suffix]={}
                for horizon in (30,90):
                    summary[label][group][suffix][str(horizon)]={}
                    for critic in ('q1','q2','q_min'):
                        selected=[r for r in raw if r['checkpoint']==label and (group=='all' or group in r['groups']) and r['suffix']==suffix and r['horizon']==horizon and r['critic']==critic]
                        c=sum(r['concordant'] for r in selected);d=sum(r['discordant'] for r in selected)
                        rho=[r['spearman'] for r in selected if r['spearman'] is not None]
                        summary[label][group][suffix][str(horizon)][critic]=dict(states=len(selected),concordant=c,discordant=d,
                              ties=sum(r['ties'] for r in selected),concordance=c/(c+d) if c+d else None,
                              mean_spearman=float(np.mean(rho)) if rho else None,top_action_agreement=sum(r['top_action_agreement'] for r in selected))
    return raw,summary


def values(root):
    saved=load(6000);replay=saved['replay']['arrays'];models=agents(root)
    models={k:v for k,v in models.items() if k!='B5600'}
    old=json.loads((BASELINE/'values_rebuilt/paired_vertical_rollouts.json').read_text())
    resets=rows(LOGS/'p0_audit/resets.jsonl');env=env_for(saved,training_course=False,pool_split='train')
    results=[];new_rollouts=0
    try:
        for prior in old:
            index=prior['replay_index'];observation,reset,error=reconstruct_state(env,index,replay,resets)
            assert error==0
            initial=observation.copy();state=torch.as_tensor(preprocess_state(observation)).reshape(1,-1)
            delta=world_goal_delta(replay['state'][index:index+1],[8,0,1])[0]
            distance=float(np.linalg.norm(delta));height=float(-delta[2])
            groups=(['near'] if distance<1.2 else [])+(['far'] if distance>=4 else [])+(['above'] if height>.15 else [])+(['below'] if height<-.15 else [])
            common_cache=[]
            for z in ZGRID:
                first=np.array([*prior['actor_action'][:2],z],dtype=np.float32)
                restored,_,again=reconstruct_state(env,index,replay,resets);np.testing.assert_array_equal(restored,initial);assert again==0
                common_cache.append(finite_rollout(env,restored,first,models['A5000'].actor));new_rollouts+=1
            record=dict(index=index,groups=groups,distance=distance,height_error=height,initial_xy=prior['actor_action'][:2],
                        reconstructed_observation_max_error=error,method='Fresh reset + recorded full real-action prefix for every physical branch',scans={},actor_actual={})
            for label,agent in models.items():
                with torch.no_grad():
                    actual=agent.actor(state);q1,q2=agent.critic(state,actual)
                record['actor_actual'][label]=dict(action=actual[0].numpy().tolist(),q1=float(q1),q2=float(q2),q_min=float(torch.minimum(q1,q2)))
                values=[]
                for z_index,z in enumerate(ZGRID):
                    first=np.array([*prior['actor_action'][:2],z],dtype=np.float32)
                    with torch.no_grad():q1,q2=agent.critic(state,torch.as_tensor(first).reshape(1,3))
                    if label in ('A5000','B5200'):
                        assert fingerprint(agent.actor)==fingerprint(models['A5000'].actor)
                        own=common_cache[z_index]
                    elif label=='A6000':
                        existing=next(s for s in prior['scans'] if s['normalized_z']==float(z) and s['suffix']=='frozen_actor_6k')
                        np.testing.assert_array_equal(existing['xy_action'],prior['actor_action'][:2])
                        own=existing  # Exact same D2 first action, plant/reward and suffix.
                    else:
                        restored,_,again=reconstruct_state(env,index,replay,resets);np.testing.assert_array_equal(restored,initial);assert again==0
                        own=finite_rollout(env,restored,first,agent.actor);new_rollouts+=1
                    values.append(dict(normalized_z=float(z),world_vz_mps=float(z*.3),q1=float(q1),q2=float(q2),q_min=float(torch.minimum(q1,q2)),
                                       physical=dict(common=common_cache[z_index],own=own)))
                record['scans'][label]=values
            # Repeat an identical one-step branch with complete fresh reconstruction.
            first=np.array([*prior['actor_action'][:2],0.],dtype=np.float32)
            restored,_,_=reconstruct_state(env,index,replay,resets);one=finite_rollout(env,restored,first,models['A5000'].actor,horizon=1)
            restored,_,_=reconstruct_state(env,index,replay,resets);two=finite_rollout(env,restored,first,models['A5000'].actor,horizon=1)
            assert one==two;record['one_step_repeat_exact']=True
            results.append(record);write(root/'paired_values.json',results)
            print(f'[D3 value] index={index} groups={groups} new_branches={new_rollouts}',flush=True)
        raw,summary=ranking_summary(results);write(root/'ranking_pairs.json',raw);write(root/'ranking_summary.json',summary)
        write(root/'physical_proof.json',dict(states=len(results),new_full_branches=new_rollouts,reused_A6k_D2_branches=60,
              common_suffix='Original frozen5k Actor',first_xy='Original D2 A6k XY for exactly one upper interval',
              state_reconstruction_exact=True,one_step_repeat_exact=True,save_restore_used=False,
              horizon_upper_steps=[30,90],horizon_seconds=[2,6],infinite_horizon_truth_claim=False))
    finally:env.close()


class LowerCollector:
    def __init__(self,env):
        self.env=env;self.original=env.base_env.step;self.samples=[];self.active=False
        def lower(action):
            output=self.original(action)
            if self.active:
                info=output[4]
                self.samples.append(dict(position=env.pos[0].tolist(),velocity=env.vel[0].tolist(),
                    reference_position=env.reference_position.tolist(),distance=float(info['goal_distance']),
                    clearance=float(info['min_tree_clearance']),collision=bool(info['collision'])))
            return output
        env.base_env.step=lower
    def close(self):self.env.base_env.step=self.original


def eval_episode(env,actor,observation,collector):
    initial=env.pos[0].copy();collector.samples=[];collector.active=True
    upper=[];total=0.
    while True:
        with torch.no_grad():action=actor(torch.as_tensor(preprocess_state(observation)).reshape(1,-1))[0].numpy()
        observation,reward,terminated,truncated,info=env.step(action.reshape(1,3));total+=reward
        upper.append(dict(action=action.tolist(),position=env.pos[0].tolist(),velocity=env.vel[0].tolist(),reference_tracking_error=float(info['reference_tracking_error_mean'])))
        if terminated or truncated:break
    collector.active=False
    lower=collector.samples;distance=np.array([s['distance'] for s in lower]);speed=np.linalg.norm([s['velocity'] for s in lower],axis=1)
    error=np.linalg.norm(np.array([s['position'] for s in lower])-np.array([s['reference_position'] for s in lower]),axis=1)
    arrivals={}
    for radius in (.2,.3,.5,1.2):
        ids=np.flatnonzero(distance<radius)
        arrivals[str(radius)]=dict(reached=bool(len(ids)),first_time_s=(int(ids[0])+1)/120 if len(ids) else None,
                                  first_speed=float(speed[ids[0]]) if len(ids) else None,time_inside_s=len(ids)/120)
    return dict(map_index=env._pool_index,map_id=env._pool_info()['map_id'],map_split='validation',
                initial_position=initial.tolist(),done_reason=info['done_reason'],success=bool(info['success']),collision=bool(info['collision']),
                minimum_goal_distance=float(min(np.linalg.norm(initial-env.TARGET_POS),distance.min())),final_goal_distance=float(distance[-1]),
                guidance_entered=bool(distance.min()<1.2),radius_arrivals=arrivals,tracking_error_mean=float(error.mean()),tracking_error_max=float(error.max()),
                actual_speed_mean=float(speed.mean()),minimum_tree_clearance=float(min(s['clearance'] for s in lower)),
                duration_s=len(lower)/120,raw_return=total,vertical_action_mean=float(np.mean([s['action'][2] for s in upper])),
                vertical_saturation_fraction=float(np.mean([abs(s['action'][2])>=.999 for s in upper])),upper=upper,lower=lower)


def evaluation(root):
    models=agents(root);env=env_for(load(6000),training_course=False,pool_split='validation');collector=LowerCollector(env)
    results={};(root/'validation_traces').mkdir(exist_ok=True)
    try:
        for label in ('A5000','A6000','B5325','B5600','B6000'):
            results[label]={}
            for task in ('L0','original16m'):
                episodes=[]
                for index in range(20):
                    observation,_=env.reset(seed=index)
                    if task=='L0':
                        path=env.pool.path_for('validation',index,0)
                        remaining=np.r_[np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1)[::-1])[::-1],0.]
                        node=int(np.argmin(np.abs(remaining-.8)));assert .35<=remaining[node]<1.2
                        observation=relocate_reset(env,path[node])
                    episode=eval_episode(env,models[label].actor,observation,collector)
                    upper=episode.pop('upper');lower=episode.pop('lower')
                    np.savez_compressed(root/'validation_traces'/f'{label}_{task}_map{index:02d}.npz',
                         lower_position=np.array([s['position'] for s in lower]),lower_velocity=np.array([s['velocity'] for s in lower]),
                         lower_reference_position=np.array([s['reference_position'] for s in lower]),lower_goal_distance=np.array([s['distance'] for s in lower]),
                         lower_collision=np.array([s['collision'] for s in lower]),upper_action=np.array([s['action'] for s in upper]),
                         upper_position=np.array([s['position'] for s in upper]),upper_velocity=np.array([s['velocity'] for s in upper]))
                    episode.update(actor_sha256=fingerprint(models[label].actor));episodes.append(episode)
                    results[label][task]=episodes;write(root/'validation.json',results)
                    print(f'[D3 eval] {label} {task} map={index} {episode["done_reason"]} min={episode["minimum_goal_distance"]:.3f}',flush=True)
        assert sum(e['success'] for e in results['A5000']['L0'])==11
        assert sum(e['success'] for e in results['A6000']['L0'])==0
        # 5200 is exactly the common initial Actor, not an independent policy.
        for label in ('B5200',):
            assert fingerprint(models[label].actor)==fingerprint(models['A5000'].actor)
            results[label]=copy.deepcopy(results['A5000'])
            for episodes in results[label].values():
                for e in episodes:e['reused_identical_actor_from']='A5000'
        write(root/'validation.json',results)
    finally:collector.close();env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=('values','evaluation'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args();globals()[args.part](args.output)


if __name__=='__main__':main()
