"""RV1 MSGP real-environment interface smoke; not a performance experiment."""
import argparse,copy,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.checkpoint import capture,restore_environment
from ms_td3 import SCHEMA,ObservationEncoder,GeometricTD3,BalancedReplay

def run(steps,out,preflight=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(43);random.seed(43);rng=np.random.default_rng(43)
    cfg=load_config();td=copy.deepcopy(cfg['td3'])
    if preflight:td['batch_size']=32;td['update_after']=32;td['warmup']=32
    pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=True)
    encoder=ObservationEncoder(env.lidar.config);agent=GeometricTD3(td);replay=BalancedReplay(steps,td['success_replay_size'],td['success_replay_fraction'])
    initial_actor=[x.detach().clone() for x in agent.actor.parameters()];initial_critic=[x.detach().clone() for x in agent.critic.parameters()]
    outcomes={};timings=[];updates=[];start=time.perf_counter();episode_id=0;episode_return=0.;step=0
    metadata={'schema':SCHEMA,'mode':'preflight' if preflight else 'smoke','seed':43,'steps':steps,'config':cfg,'effective_td3':td,'pool_fingerprint':pool.fingerprint,'replay_capacity_override':steps,'note':'engineering test only; no navigation performance claim'}
    (out/'config.json').write_text(json.dumps(metadata,indent=2,default=str))
    try:
        raw,info=env.reset(seed=43);s=encoder(raw)
        for step in range(1,steps+1):
            t0=time.perf_counter()
            if step<=td['warmup']:a=rng.uniform(-1,1,3)
            else:
                sigma=max(td['exploration_end'],td['exploration_start']-(td['exploration_start']-td['exploration_end'])*step/td['noise_decay_steps'])
                a=agent.action(s)+rng.normal(0,sigma,3)
            a=project(torch.as_tensor(a,dtype=torch.float32)).numpy();before=env.lidar.cache.timestamp_s
            raw2,reward,term,trunc,info=env.step(a);after=env.lidar.cache.timestamp_s
            if after<=before or env.lidar.scan_count!=env.upper_tick+1:raise RuntimeError('scan timing/count')
            n=encoder(raw2);executed=np.asarray(info['projected_action'],np.float32)
            replay.add(s,executed,n,reward*cfg['reward_scale'],term,trunc);s=n;episode_return+=reward
            if step>=td['update_after']:
                diag=agent.update(replay)
                if not np.isfinite(diag['critic_loss']):raise FloatingPointError('nonfinite loss')
                updates.append(diag)
            if term or trunc:
                reason=info['done_reason'];outcomes[reason]=outcomes.get(reason,0)+1
                replay.finish(reason=='success',(10,20,30,40)[info['density_stage']]);env.record_episode_outcome(reason=='success',collision=info['collision'],done_reason=reason)
                episode_id+=1;episode_return=0.;raw,info=env.reset();s=encoder(raw)
            timings.append(time.perf_counter()-t0)
            if step%500==0:print(json.dumps({'step':step,'elapsed_s':round(time.perf_counter()-start,1),'updates':agent.updates,'outcomes':outcomes}),flush=True)
        torch.save({'schema':SCHEMA,'agent':agent.state(),'replay':replay,'environment':capture(env),'observation':s,'torch_rng':torch.get_rng_state(),'action_rng':rng.bit_generator.state,'step':step},out/'checkpoint.pt')
        payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if payload['schema']!=SCHEMA or payload['step']!=steps:raise RuntimeError('checkpoint schema')
        restored=GeometricTD3(td);restored.restore(copy.deepcopy(payload['agent']));replay2=payload['replay']
        if not np.array_equal(restored.action(s),agent.action(s)):raise RuntimeError('checkpoint actor mismatch')
        clone=make_course_env(pool,training=True)
        try:
            restored_obs=restore_environment(clone,payload['environment']);np.testing.assert_array_equal(encoder(restored_obs),payload['observation'])
        finally:clone.close()
        # Check identical update after independently restored replay and RNG.
        if updates:
            r1=copy.deepcopy(replay);r2=copy.deepcopy(replay2)
            torch.set_rng_state(payload['torch_rng']);d1=agent.update(r1)
            torch.set_rng_state(payload['torch_rng']);d2=restored.update(r2)
            if d1!=d2 or any(not torch.equal(a,b) for a,b in zip(agent.critic.parameters(),restored.critic.parameters())):raise RuntimeError('restored update mismatch')
        summary={'status':'PASS','steps':step,'wall_s':time.perf_counter()-start,'critic_updates_at_checkpoint':payload['agent']['updates'],'actor_updates_at_checkpoint':payload['agent']['actor_updates'],
                 'actor_parameters_changed':any(not torch.equal(p,q) for p,q in zip(restored.actor.parameters(),initial_actor)),
                 'critic_parameters_changed':any(not torch.equal(p,q) for p,q in zip(restored.critic.parameters(),initial_critic)),
                 'replay_total':replay.main.total,'success_sizes':{str(k):v.size for k,v in replay.success.items()},'outcomes':outcomes,
                 'checkpoint_replay_rng_environment_restore':True,'next_update_checked':bool(updates),
                 'step_p50_ms':float(np.quantile(timings,.5)*1000),'step_p95_ms':float(np.quantile(timings,.95)*1000),'note':'includes physics and TD3 updates; throughput, not control latency'}
        if updates and not(summary['actor_parameters_changed'] and summary['critic_parameters_changed']):raise RuntimeError('network parameters unchanged')
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if not 1<=a.steps<=6000:raise ValueError('smoke limited to 6000 transitions')
    run(a.steps,a.out,a.preflight)
