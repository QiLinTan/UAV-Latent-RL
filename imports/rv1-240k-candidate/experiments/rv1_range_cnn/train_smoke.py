"""Bounded <=10k interface smoke, not a navigation-performance experiment."""
import os,sys,json,copy,hashlib,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import load_config,ROOT
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from .environment import make_course_env
from .lidar import LidarConfig
from .td3 import RangeTD3
from .replay import SuccessReplay
from .provenance import contract,sources,digest
from .checkpoint import capture,restore_environment,rng_state,restore_rng,atomic_save,validate

def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def write(path,x):Path(path).write_text(json.dumps(plain(x),indent=2,allow_nan=False))
def append(stream,x):stream.write(json.dumps(plain(x),allow_nan=False)+'\n');stream.flush()

def require_gate(out):
    gate=json.loads((out/'geometry_gate.json').read_text())
    if gate['status']!='PASS_GEOMETRY_COST':raise RuntimeError('geometry/cost gate failed')
    frozen=json.loads((out/'source_hashes_geometry.json').read_text());current=sources()
    for key,value in frozen.items():
        if any(key.endswith('/'+f) for f in ('lidar.py','codec.py','environment.py','benchmark.py','audit_geometry.py')) and current[key]!=value:raise RuntimeError('geometry source changed after audit')
    suites=ET.parse(out/'all_tests.xml').getroot()
    if any(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites.iter('testsuite')):raise RuntimeError('unit/integration tests failed')

def inference_benchmark(agent):
    rows=[]
    with torch.inference_mode():
        for batch in (1,32,256):
            sample={'state':torch.zeros(batch,18),'image':torch.zeros(batch,3,32,128)}
            for _ in range(10):agent.actor(sample)
            times=[];encoder=[]
            for _ in range(40):
                start=perf_counter();agent.actor(sample);times.append(perf_counter()-start)
                start=perf_counter();agent.actor.encoder(sample['image']);encoder.append(perf_counter()-start)
            rows.append(dict(device='cpu',batch=batch,actor_ms_median=1000*float(np.median(times)),actor_ms_p95=1000*float(np.quantile(times,.95)),encoder_ms_median=1000*float(np.median(encoder))))
    return dict(cuda_available=torch.cuda.is_available(),gpu_measured=False,cpu_threads=torch.get_num_threads(),measurements=rows)

def run(out,steps=6000):
    if not 0<steps<=10000:raise ValueError('Step3 authorization caps smoke at 10k transitions')
    out=Path(out);require_gate(out);run=out/'cnn_smoke_seed43'
    run.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(43);rng=np.random.default_rng(43)
    cfg=load_config();pool=ForestPool(ROOT/cfg['forest_pool_path']);catalog=json.loads((out/'challenge_catalog.json').read_text())
    c=contract(LidarConfig(),catalog['fingerprint']);td3_config=copy.deepcopy(cfg['td3'])
    agent=RangeTD3(td3_config,c);replay=SuccessReplay(c,capacity=steps,success_capacity=td3_config['success_replay_size'],fraction=td3_config['success_replay_fraction'])
    env=make_course_env(pool);obs,info=env.reset(seed=43);episode_id=0;episode_reward=0.;updates=[];checkpoints=[]
    initial_actor={k:p.detach().clone() for k,p in agent.actor.encoder.named_parameters()}
    initial_critic={k:p.detach().clone() for k,p in agent.critic.encoder.named_parameters()}
    metadata=dict(contract=c,seed=43,total_steps=steps,hyperparameters=td3_config,
        replay_capacity_smoke_override=steps,success_replay_capacity=td3_config['success_replay_size'],
        forest_pool_fingerprint=pool.fingerprint,challenge_catalog_fingerprint=catalog['fingerprint'],
        training_data='original Forest TRAIN pool, unchanged start/obstacle curriculum; no challenge validation/test data',
        goal='interface/gradient/restore validation only, not proof of learned navigation or CNN perception benefit',
        source_hashes=sources(),model_parameters={name:sum(p.numel() for p in getattr(agent,name).parameters()) for name in ('actor','critic')},
        encoder_parameters=sum(p.numel() for p in agent.actor.encoder.parameters()),spatial_shape=agent.actor.encoder.spatial_shape,
        inference=inference_benchmark(agent),frozen_config=cfg,
        differences_from_dv0=['raw single-scale image encoder instead of 96-vector MLP','dense scan at 15Hz vs dv0 legacy 96 at 120Hz',
            'bounded smoke main Replay capacity, not a long-run ablation','independent Actor/Critic encoders trained by TD3; two Critic heads share one encoder'])
    write(run/'metadata.json',metadata);start_wall=perf_counter();update_durations=[];severe_count=0;episode_outcomes=[]

    def save(step,tag=''):
        replay_file=run/f'replay_step{step:06d}{tag}.pt';replay.save(replay_file)
        payload=dict(schema=c['schema'],format='rv1_training_smoke_v1',contract=c,agent=agent.state(),
            environment=capture(env),rng=rng_state(),action_rng=rng.bit_generator.state,
            trainer=dict(step=step,episode_id=episode_id,episode_reward=episode_reward),observation=obs,
            replay_path=str(replay_file),replay_sha256=hashlib.sha256(replay_file.read_bytes()).hexdigest())
        checkpoint=run/f'checkpoint_step{step:06d}{tag}.pt';atomic_save(checkpoint,payload)
        checkpoints.append(dict(step=step,path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        return payload,replay_file

    try:
        with (run/'transitions.jsonl').open('w') as trace,(run/'updates.jsonl').open('w') as learning,(run/'episodes.jsonl').open('w') as episodes:
            for step in range(1,steps+1):
                before_time=env.lidar.cache.timestamp_s
                if step<=td3_config['warmup']:raw=rng.uniform(-1,1,3)
                else:
                    raw=agent.action(obs)+rng.normal(0,max(td3_config['exploration_end'],td3_config['exploration_start']-(td3_config['exploration_start']-td3_config['exploration_end'])*step/td3_config['noise_decay_steps']),3)
                action=project(torch.as_tensor(raw,dtype=torch.float32)).numpy()
                next_obs,reward,t,u,info=env.step(action)
                if not np.isfinite(next_obs['state']).all() or not np.isfinite(next_obs['range_m']).all():raise FloatingPointError('nonfinite observation')
                if env.lidar.scan_count!=env.upper_tick+1:raise RuntimeError('dense scan repeated or missing')
                replay.add(obs,info['projected_action'],next_obs,reward*cfg['reward_scale'],t,u,episode_id=episode_id,timestamps=(before_time,env.lidar.cache.timestamp_s))
                tick=env.last_ticks;velocity=np.array([x['velocity'] for x in tick]);command=np.array([x['command_world'] for x in tick])
                append(trace,dict(step=step,episode_id=episode_id,state=obs['state'],raw_action=raw,accepted_action=info['projected_action'],
                    accepted_world=info['accepted_world_command'],reward_raw=reward,reward_terms={k:v for k,v in info.items() if k.startswith('reward/')},
                    goal_distance=info['goal_distance'],clearance=info['min_tree_clearance'],done_reason=info['done_reason'],terminated=t,truncated=u,
                    density=info['density_stage'],start_level=info['start_curriculum_episode_level'],map_index=info['map_pool_index'],
                    geometry_sha256=info['geometry_sha256'],velocity_rmse=float(np.sqrt(np.mean(np.sum((velocity-command)**2,axis=1)))),
                    roll_pitch_peak=float(abs(np.array([x['rpy'][:2] for x in tick])).max()),
                    rpm_max=float(np.array([x['rpm'] for x in tick]).max()),allocation_saturated_fraction=float(np.mean([x['allocation_saturated'] for x in tick])),
                    lower_ticks=len(tick),sensor_timestamp_s=env.lidar.cache.timestamp_s))
                obs=next_obs;episode_reward+=reward
                if step>=td3_config['update_after']:
                    ts=perf_counter();d=agent.update(replay);update_durations.append(perf_counter()-ts);d['step']=step
                    # Encoder gradients must actually be finite and connected.
                    d['critic_encoder_gradient_finite']=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in agent.critic.encoder.parameters())
                    d['actor_encoder_gradient_finite']=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in agent.actor.encoder.parameters()) if d['actor_updated'] else None
                    if not d['critic_encoder_gradient_finite'] or d['actor_encoder_gradient_finite'] is False:raise FloatingPointError('disconnected/nonfinite encoder gradient')
                    append(learning,d);updates.append(d)
                    bad=d['q_abs_max']>cfg['diagnostics']['severe_q_abs'] or d['critic_loss']>cfg['diagnostics']['severe_loss']
                    severe_count=severe_count+1 if bad else 0
                    if severe_count>=cfg['diagnostics']['severe_consecutive_updates']:raise FloatingPointError('predeclared severe learning condition')
                if t or u:
                    success=info['done_reason']=='success';density=(10,20,30,40)[info['density_stage']]
                    replay.finish(success,density);env.record_episode_outcome(success,collision=info['collision'],done_reason=info['done_reason'])
                    row=dict(episode_id=episode_id,end_step=step,done_reason=info['done_reason'],reward_raw=episode_reward,
                             start_source=info['start_curriculum_source'],start_level=info['start_curriculum_episode_level'],density=info['density_stage'],map_index=info['map_pool_index'])
                    append(episodes,row);episode_outcomes.append(row);episode_id+=1;episode_reward=0.;obs,info=env.reset()
                if step in (2000,5000,steps):
                    diag=dict(step=step,critic_updates=agent.updates,actor_updates=agent.actor_updates,
                        replay=replay.diagnostics(),networks=agent.diagnostics(replay),elapsed_wall_s=perf_counter()-start_wall,
                        learning_active=step>=td3_config['update_after'])
                    write(run/f'diagnostic_step{step:06d}.json',diag);print(json.dumps(diag),flush=True)
                    payload,replay_path=save(step)
            # Restore full networks/optimizers/Replay/current Full episode, then
            # perform an identical diagnostic next update from saved RNG. No new
            # environment transition and no extension of the authorized budget.
            restored_replay=SuccessReplay.load(replay_path,c);restored_agent=RangeTD3(td3_config,c);validate(payload,c)
            restored_agent.restore(payload['agent']);clone=make_course_env(pool)
            try:
                reobs=restore_environment(clone,payload['environment'])
                for key in obs:np.testing.assert_array_equal(obs[key],reobs[key])
                restore_rng(payload['rng']);a=agent.update(replay)
                restore_rng(payload['rng']);b=restored_agent.update(restored_replay)
                if a!=b:raise RuntimeError('checkpoint next-update metrics mismatch')
                for name in ('actor','critic'):
                    for x,y in zip(getattr(agent,name).parameters(),getattr(restored_agent,name).parameters()):
                        if not torch.equal(x,y):raise RuntimeError('checkpoint next-update parameter mismatch')
            finally:clone.close()
        changes={}
        for name,original in [('actor',initial_actor),('critic',initial_critic)]:
            changes[name+'_encoder_l2_delta']=float(torch.sqrt(sum(((p-original[k])**2).sum() for k,p in getattr(agent,name).encoder.named_parameters())))
        gates=dict(replay_transitions=replay.main.total==steps,critic_updates=payload['agent']['updates']==max(0,steps-td3_config['update_after']+1),
            actor_updates=payload['agent']['actor_updates']==payload['agent']['updates']//td3_config['policy_delay'],
            finite_encoder_gradients=all(x['critic_encoder_gradient_finite'] and x['actor_encoder_gradient_finite'] is not False for x in updates),
            encoder_parameters_changed=all(v>0 for v in changes.values()),checkpoint_optimizer_replay_physics_restore=True,
            no_heldout_training_data=True,within_10k_budget=steps<=10000)
        summary=dict(status='PASS_CNN_TD3_SMOKE' if all(gates.values()) else 'FAIL_CNN_TD3_SMOKE',gates=gates,steps=steps,
            critic_updates=payload['agent']['updates'],actor_updates=payload['agent']['actor_updates'],
            diagnostic_only_extra_update=1,extra_environment_transitions=0,
            wall_seconds=perf_counter()-start_wall,training_update_ms_median=float(np.median(update_durations)*1000),
            training_update_ms_p95=float(np.quantile(update_durations,.95)*1000),encoder_changes=changes,
            replay=replay.diagnostics(),episode_done_counts={k:sum(x['done_reason']==k for x in episode_outcomes) for k in sorted(set(x['done_reason'] for x in episode_outcomes))},
            checkpoints=checkpoints,final_checkpoint=str(run/f'checkpoint_step{steps:06d}.pt'),contract=c,
            conclusion='Only numerical/interface health established. CNN representations and navigation benefit remain unvalidated.')
        write(run/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
        if not all(gates.values()):raise RuntimeError('smoke gates failed')
    except BaseException as error:
        write(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),last_step=locals().get('step'),contract=c))
        try:save(locals().get('step',0),'_failure')
        except BaseException:pass
        raise
    finally:env.close()

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 6000)
