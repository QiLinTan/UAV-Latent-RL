"""Automatic Gate B -> short RL verification -> continuous 240k exploration."""
import argparse,copy,gzip,hashlib,json,os,random,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from envs.map_atlas.forest_pool import ForestPool
from .task_factory import ROOT,EXP,load_config,data_contract,digest,source_hashes,make_env
from .rl.td3_adapter import TD3,project
from .rl.versioned_replay import VersionedReplay
from .rl.checkpoint import capture_episode,restore_episode,atomic_save,validate,rng_state,restore_rng
from .evaluate import evaluate


def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));os.replace(temp,path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=load_config();pool=ForestPool(ROOT/config['forest_pool_path']);contract=data_contract(config,pool)
    gate=json.loads(args.gate_b.read_text())
    if not gate.get('passed') or gate['contract']!=contract:raise RuntimeError('Gate B missing, failed or stale')
    random.seed(43);np.random.seed(43);torch.manual_seed(43)
    output=args.output.resolve()
    if EXP/'outputs' not in output.parents:raise ValueError('output outside independent experiment')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoints','replay','evaluations','diagnostics','traces'): (output/name).mkdir()
    write(output/'configuration.json',config);write(output/'contract.json',contract)
    write(output/'source_hashes.json',source_hashes());write(output/'gate_b.json',gate)
    agent=TD3(config['td3']);r=VersionedReplay(contract)
    env=make_env(config,pool=pool);obs,info=env.reset(seed=43)
    step=0;episode_id=0;episode_return=0.;episode_steps=0;episode_terms={};last_diag={};severe_count=0
    start=time.monotonic();prior_elapsed=0.;total_lower=0;phase='short_validation';stop_requested=False
    actor_initial=torch.cat([p.detach().reshape(-1) for p in agent.actor.parameters()]).clone()
    critic_initial=torch.cat([p.detach().reshape(-1) for p in agent.critic.parameters()]).clone()
    counters={};window=[];history_episodes=[];last_saved=None
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu',weights_only=False);validate(payload,contract)
        replay_path=args.resume.parent.parent/payload['replay_path']
        if sha(replay_path)!=payload['replay_sha256']:raise ValueError('Replay file hash mismatch')
        r.restore(torch.load(replay_path,map_location='cpu',weights_only=False));agent.restore(payload['agent'])
        obs=restore_episode(env,payload['environment']);state=payload['trainer']
        step=state['step'];episode_id=state['episode_id'];episode_return=state['episode_return'];episode_steps=state['episode_steps']
        episode_terms=state['episode_terms'];phase=state['phase'];total_lower=state['total_lower'];counters=state['counters']
        prior_elapsed=state['elapsed_wall_s'];history_episodes=state['history_episodes'];severe_count=state['severe_count']
        actor_initial=state['actor_initial'];critic_initial=state['critic_initial'];restore_rng(payload['rng'])
        write(output/'resume_source.json',dict(checkpoint=str(args.resume),step=step))
    def handler(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    def status(state='RUNNING',error=None):
        return dict(status=state,phase=phase,step=step,budget=config['total_steps'],seed=43,
            elapsed_wall_s=prior_elapsed+time.monotonic()-start,lower_ticks=total_lower,simulation_s=total_lower/120,
            updates=agent.updates,actor_updates=agent.actor_updates,replay=r.diagnostics(),
            curriculum=env.course_info(),outcomes=counters,last_learning=last_diag,last_checkpoint=last_saved,error=error,
            pid=os.getpid(),run=str(output),config_sha256=digest(config))
    def save(label=None):
        nonlocal last_saved
        label=label or f'step{step:06d}'
        replay=output/'replay'/f'replay_{label}.pt';atomic_save(replay,r.state())
        checkpoint=output/'checkpoints'/f'training_state_{label}.pt'
        state=dict(step=step,episode_id=episode_id,episode_return=episode_return,episode_steps=episode_steps,
            episode_terms=episode_terms,phase=phase,total_lower=total_lower,counters=counters,
            elapsed_wall_s=prior_elapsed+time.monotonic()-start,history_episodes=history_episodes[-100:],
            severe_count=severe_count,actor_initial=actor_initial,critic_initial=critic_initial)
        atomic_save(checkpoint,dict(format='dv0_training_v1',contract=contract,agent=agent.state(),
            environment=capture_episode(env),trainer=state,rng=rng_state(),
            replay_path=str(replay.relative_to(output)),replay_sha256=sha(replay)))
        last_saved=str(checkpoint);write(output/'latest_checkpoint.json',dict(path=str(checkpoint),step=step,sha256=sha(checkpoint),replay=str(replay)))
    log=(output/'learning.jsonl').open('a',buffering=1)
    episodes=(output/'episodes.jsonl').open('a',buffering=1)
    trace=gzip.open(output/'traces/upper_and_lower.jsonl.gz','at',compresslevel=1)
    write(output/'status.json',status())
    try:
        while step<config['total_steps']:
            if stop_requested:
                save('interrupted_step'+str(step));write(output/'status.json',status('STOPPED_SIGNAL'));return
            raw=agent.action(obs) if step>=config['td3']['warmup'] else np.random.uniform(-1,1,3)
            if step>=config['td3']['warmup']:
                noise=config['td3']['exploration_start']+(config['td3']['exploration_end']-config['td3']['exploration_start'])*min(step/config['td3']['noise_decay_steps'],1)
                raw=raw+np.random.normal(0,noise,3)
            next_obs,reward,terminated,truncated,info=env.step(raw)
            done=terminated or truncated
            r.push(obs,info['projected_action'],reward*config['reward_scale'],next_obs,done,
                   info['accepted_world_command'],info['lower_steps_executed'])
            step+=1;episode_steps+=1;episode_return+=reward;total_lower+=info['lower_steps_executed']
            for k,v in info.items():
                if k.startswith('reward/'):episode_terms[k]=episode_terms.get(k,0.)+v
            trace.write(json.dumps(clean(dict(step=step,episode=episode_id,raw_action=raw,
                projected_action=info['projected_action'],reward=reward,goal_distance=info['goal_distance'],
                density=info['active_tree_count'],start_level=env._start_level,map_index=env._pool_index,
                reward_terms={k:v for k,v in info.items() if k.startswith('reward/')},ticks=env.last_ticks)),allow_nan=False)+'\n')
            for t in env.last_ticks:window.append(t)
            obs=next_obs
            if step>=config['td3']['update_after']:
                last_diag=agent.update(r);log.write(json.dumps(dict(step=step,**last_diag),allow_nan=False)+'\n')
                severe=last_diag['q_abs_max']>config['diagnostics']['severe_q_abs'] or last_diag['td_target_abs_max']>config['diagnostics']['severe_q_abs'] or last_diag['critic_loss']>config['diagnostics']['severe_loss']
                severe_count=severe_count+1 if severe else 0
                if severe_count>=config['diagnostics']['severe_consecutive_updates']:raise FloatingPointError('frozen severe value/loss threshold persisted for 100 updates')
            if done:
                row=dict(step=step,episode=episode_id,upper_steps=episode_steps,return_raw=episode_return,
                    done_reason=info['done_reason'],success=info['success'],collision=info['collision'],
                    final_distance=info['goal_distance'],course=env.course_info(),reward_terms=episode_terms)
                episodes.write(json.dumps(clean(row),allow_nan=False)+'\n');history_episodes.append(row);history_episodes=history_episodes[-100:]
                counters[info['done_reason']]=counters.get(info['done_reason'],0)+1
                r.finish(info['success'],info['active_tree_count']);env.record_episode_outcome(info['success'],info['collision'],info['done_reason'])
                obs,info=env.reset();episode_id+=1;episode_return=0.;episode_steps=0;episode_terms={}
            if step%500==0 or step in (2000,5000,10000):
                batch=r.sample(min(256,r.main.size))[0]
                diag=agent.diagnostics(batch)
                if diag['parameter_abs_max']>config['diagnostics']['parameter_abs']:raise FloatingPointError('parameter magnitude hard threshold')
                a_now=torch.cat([p.detach().reshape(-1) for p in agent.actor.parameters()]);q_now=torch.cat([p.detach().reshape(-1) for p in agent.critic.parameters()])
                w=window;window=[]
                diag.update(step=step,actor_parameter_delta_l2=float(torch.linalg.vector_norm(a_now-actor_initial)),
                    critic_parameter_delta_l2=float(torch.linalg.vector_norm(q_now-critic_initial)),updates=agent.updates,
                    actor_updates=agent.actor_updates,replay=r.diagnostics(),last_learning=last_diag,
                    recent_outcomes={reason:sum(e['done_reason']==reason for e in history_episodes) for reason in counters},
                    velocity_error_rmse=float(np.sqrt(np.mean([np.sum((np.asarray(t['velocity'])-t['command_world'])**2) for t in w]))),
                    height_min=min(t['position'][2] for t in w),height_max=max(t['position'][2] for t in w),
                    attitude_peak=max(max(abs(np.asarray(t['rpy'])[:2])) for t in w),
                    force_saturation_fraction=float(np.mean([t['force_saturated'] for t in w])),
                    rpm_min=min(min(t['rpm']) for t in w),rpm_max=max(max(t['rpm']) for t in w),
                    allocation_saturation_fraction=float(np.mean([t['allocation_saturated'] for t in w])))
                risks=[]
                if max(diag['raw_axis_boundary_rate'])>.95:risks.append('persistent axis boundary risk; observation only')
                if diag['policy_q_mean']<diag['zero_action_q_mean']:risks.append('critic ranks zero above actor; observation only')
                diag['learning_risks']=risks
                write(output/'diagnostics'/f'diagnostic_step{step:06d}.json',diag)
                if step==10000:
                    if agent.updates<5000 or agent.actor_updates<2400 or diag['actor_parameter_delta_l2']<=0 or diag['critic_parameter_delta_l2']<=0:
                        raise RuntimeError('short RL mechanism validation failed')
                    phase='formal_exploration'
                    write(output/'short_validation.json',dict(passed=True,step=step,diagnostic=diag,
                        continuation='same replay, networks, targets, optimizers and RNG; no reinitialization'))
                print(json.dumps(clean(dict(step=step,phase=phase,updates=agent.updates,course=env.course_info(),risks=risks))),flush=True)
            if step%100==0:write(output/'status.json',status())
            if step in (2000,5000,10000,15000,20000) or step>20000 and step%10000==0:
                save();trace.flush();write(output/'status.json',status())
            if step%10000==0:
                result=evaluate(agent,config,pool,step,env.start_curriculum.level,env.obstacle_curriculum.stage,complete=step%20000==0)
                write(output/'evaluations'/f'evaluation_step{step:06d}.json',result)
        save('final_step'+str(step));write(output/'status.json',status('COMPLETE'))
    except BaseException as error:
        trace.flush()
        detail=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),status=status('FAILED'))
        write(output/'failure.json',detail)
        try:save('failure_step'+str(step))
        except Exception as saving:write(output/'failure_save_error.json',dict(error=repr(saving)))
        write(output/'status.json',status('FAILED',repr(error)));raise
    finally:
        trace.close();log.close();episodes.close();env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gate-b',type=Path,required=True);parser.add_argument('--resume',type=Path)
    run(parser.parse_args())
