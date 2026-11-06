"""Versioned, explicit-device rv1 trainer. Outputs are always a new run directory."""
import subprocess
import argparse, copy, hashlib, json, os, random, resource, time, traceback, sys, shutil
from pathlib import Path
sys.path[:0]=["/opt/uav-training-20261103/rv2_project", str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config,digest
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.environment import make_course_env
from learnable_agent import StaticLearnableTD3 as RangeTD3
from experiments.rv1_range_cnn.replay import SuccessReplay
from experiments.rv1_range_cnn.provenance import SCHEMA,contract,sources,validate_contract
from experiments.rv1_range_cnn.checkpoint import capture,restore_environment,rng_state,restore_rng,atomic_save

FORMAT='nf_staticlearnable_128_scratch_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def log(path,record):
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()

def save(run,step,agent,replay,env,obs,rng,episode_id,episode_reward,severe,c,device,training_spec):
    rp=run/('replay_%06d.pt'%step); cp=run/('checkpoint_%06d.pt'%step)
    if rp.exists() or cp.exists():raise FileExistsError('checkpoint already exists')
    tmp=rp.with_suffix('.pt.tmp')
    replay.save(tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,rp)
    state={'format':FORMAT,'schema':SCHEMA,'contract':c,'training_spec':training_spec,'device':str(device),
           'agent':agent.state(),'environment':capture(env),'observation':obs,
           'rng':rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
           'action_rng':copy.deepcopy(rng.bit_generator.state),
           'trainer':{'step':step,'episode_id':episode_id,'episode_reward':episode_reward,'severe':severe},
           'replay_file':rp.name,'replay_sha256':sha(rp)}
    atomic_save(cp,state)
    return cp

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after',type=int,help='bounded interruption for testing; never extends the 240k budget')
    parser.add_argument('--checkpoint-interval',type=int,default=5000)
    args=parser.parse_args();run=args.run_dir.resolve();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.checkpoint_interval<1:raise ValueError('invalid checkpoint interval')
    if args.resume is None:run.mkdir(parents=True,exist_ok=False)
    elif not run.is_dir():raise FileNotFoundError('run directory missing')
    torch.set_num_threads(2);cfg=load_config();td=cfg['td3'];total=cfg['total_steps']
    end=total if args.stop_after is None else args.stop_after
    if not 1<=end<=total:raise ValueError('stop-after outside configured budget')
    catalog=json.loads((ROOT/'evidence/challenge_catalog.json').read_text()); fingerprint=catalog['fingerprint']
    without={k:v for k,v in catalog.items() if k!='fingerprint'}
    if digest(without)!=fingerprint:raise ValueError('challenge catalog fingerprint mismatch')
    c=contract(LidarConfig(),fingerprint);pool=ForestPool(ROOT/cfg['forest_pool_path'])
    architecture_files=[Path(__file__).resolve(),Path(__file__).resolve().parent/'learnable_agent.py',Path(__file__).resolve().parent/'config.json',Path(__file__).resolve().parent.parent/'nf_gated_main_v2/nf_main/models.py',Path(__file__).resolve().parent.parent/'nf_gated_main_v2/nf_main/agent.py']
    architecture_sha={p.name:sha(p) for p in architecture_files}
    gate=json.loads((Path(__file__).resolve().parent/'engineering_gate.json').read_text())
    if gate.get('status')!='PASS':raise RuntimeError('engineering gate has not passed')
    frozen_path=Path(__file__).resolve().parent/'training_config.json'
    if cfg!=json.loads(frozen_path.read_text()):raise RuntimeError('formal training config changed since engineering audit')
    training_spec=digest({'full_config_sha256':sha(frozen_path),'seed':cfg['seed'],'total_steps':total,'diagnostics':cfg['diagnostics'],'td3':td,'pool_fingerprint':pool.fingerprint,'architecture_sha256':architecture_sha})
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
    rng=np.random.default_rng(cfg['seed']);agent=RangeTD3(td,c,device=device)
    initial={owner:{n:p.detach().cpu().clone() for n,p in getattr(agent,owner).named_parameters()} for owner in ('actor','critic')}
    env=make_course_env(pool);start=0;episode_id=0;episode_reward=0.;severe=0
    try:
        if args.resume:
            cp=args.resume.resolve()
            if cp.parent not in (run,run/'recovery'):raise ValueError('resume checkpoint must be in this run or its recovery directory')
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            if payload.get('format')!=FORMAT or payload.get('schema')!=SCHEMA:raise ValueError('incompatible old checkpoint')
            validate_contract(payload['contract'],c)
            if payload.get('training_spec')!=training_spec:raise ValueError('training configuration or pool changed')
            if payload['device']!=str(device):raise ValueError('resume must use original device; device migration needs validation')
            name=payload['replay_file']
            if Path(name).name!=name:raise ValueError('invalid replay path')
            rp=cp.parent/name
            if sha(rp)!=payload['replay_sha256']:raise ValueError('replay checksum mismatch')
            replay=SuccessReplay.load(rp,c);agent.restore(payload['agent'])
            obs=restore_environment(env,payload['environment'])
            for k in obs:
                if not np.array_equal(obs[k],payload['observation'][k]):raise RuntimeError('observation reconstruction mismatch: '+k)
            state=payload['trainer'];start=state['step'];episode_id=state['episode_id'];episode_reward=state['episode_reward'];severe=state['severe']
            if end<=start:raise ValueError('stop-after must exceed resumed step')
            rng.bit_generator.state=payload['action_rng'];restore_rng(payload['rng'])
            if device.type=='cuda':
                if len(payload['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA topology changed')
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
        else:
            replay=SuccessReplay(c,capacity=td['buffer_size'],success_capacity=td['success_replay_size'],fraction=td['success_replay_fraction'])
            obs,_=env.reset(seed=cfg['seed'])
            (run/'metadata.json').write_text(json.dumps({'format':FORMAT,'contract':c,'config':cfg,'source_hashes':sources(),'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda,'pool_fingerprint':pool.fingerprint,'architecture_sha256':architecture_sha,'initialization':'from_scratch'},indent=2))
        log(run/'events.jsonl',{'event':'start' if start==0 else 'resume','step':start,'end':end,'device':str(device),'time':time.time()})
        if start==0:
            initial_gates={owner:{'logits':[0.,0.],'weights':[.5,.5],'last_gradient_norm':None} for owner in ('actor','critic','actor_target','critic_target')}
            log(run/'gate_trajectory.jsonl',{'step':0,'actor_updates':0,'critic_updates':0,**initial_gates})
        begin=time.perf_counter()
        for step in range(start+1,end+1):
            before=env.lidar.cache.timestamp_s
            if step<=td['warmup']:raw=rng.uniform(-1,1,3)
            else:
                noise=max(td['exploration_end'],td['exploration_start']-(td['exploration_start']-td['exploration_end'])*step/td['noise_decay_steps'])
                raw=agent.action(obs)+rng.normal(0,noise,3)
            action=project(torch.as_tensor(raw,dtype=torch.float32)).numpy()
            nxt,reward,terminated,truncated,info=env.step(action)
            if not np.isfinite(nxt['state']).all() or not np.isfinite(nxt['range_m']).all():raise FloatingPointError('nonfinite observation')
            if env.lidar.scan_count!=env.upper_tick+1:raise RuntimeError('scan count mismatch')
            replay.add(obs,info['projected_action'],nxt,reward*cfg['reward_scale'],terminated,truncated,episode_id=episode_id,timestamps=(before,env.lidar.cache.timestamp_s))
            obs=nxt;episode_reward+=reward
            if step>=td['update_after']:
                t=time.perf_counter();d=agent.update(replay)
                bad=d['q_abs_max']>cfg['diagnostics']['severe_q_abs'] or d['critic_loss']>cfg['diagnostics']['severe_loss']
                severe=severe+1 if bad else 0
                log(run/'updates.jsonl',{'step':step,'elapsed_s':time.perf_counter()-t,**d})
                if severe>=cfg['diagnostics']['severe_consecutive_updates']:raise FloatingPointError('predeclared severe learning condition')
            if terminated or truncated:
                replay.finish(info['done_reason']=='success',(10,20,30,40)[info['density_stage']])
                env.record_episode_outcome(info['done_reason']=='success',collision=info['collision'],done_reason=info['done_reason'])
                log(run/'episodes.jsonl',{'step':step,'episode':episode_id,'reward':episode_reward,'done_reason':info['done_reason']})
                episode_id+=1;episode_reward=0.;obs,_=env.reset()
            if step%100==0:
                log(run/'metrics.jsonl',{'step':step,'steps_per_second':(step-start)/(time.perf_counter()-begin),'replay':replay.diagnostics(),'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cuda_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
                weights={}
                for owner in ('actor','critic','actor_target','critic_target'):
                    logits=getattr(agent,owner).encoder.gate.logits
                    weights[owner]={'logits':logits.detach().cpu().tolist(),'weights':logits.detach().softmax(0).cpu().tolist(),'last_gradient_norm':None if logits.grad is None else float(logits.grad.detach().norm())}
                log(run/'gate_trajectory.jsonl',{'step':step,'actor_updates':agent.actor_updates,'critic_updates':agent.updates,**weights})
                snapshot={'course':env.course_state(),'start_level':env._start_level,'density':env._density_episode_stage};scheduler=snapshot['course'].get('obstacle_scheduler',{});window=scheduler.get('successes',[])
                with torch.no_grad():
                    policy_action=agent.actor(agent.tensor_observation(obs))[0].detach().cpu().tolist()
                log(run/'course_and_policy.jsonl',{'step':step,'level':snapshot.get('start_level'),'density':snapshot.get('density'),'stage_episodes':scheduler.get('stage_episodes'),'window_n':len(window),'window_success':sum(window),'course':json.loads(json.dumps(snapshot['course'],default=lambda x:list(x) if hasattr(x,'__iter__') else str(x)).replace('Infinity','"inf"')),'actor_raw_action':policy_action,'actor_z_positive_saturated':policy_action[2]>=.99,'actor_z_negative_saturated':policy_action[2]<=-.99})
            if step%1000==0:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip() if device.type=='cuda' else None
                log(run/'resources.jsonl',{'step':step,'gpu_memory_MiB_util_percent_power_W':gpu,'host_free_bytes':shutil.disk_usage('/mnt/d').free})
            if step%args.checkpoint_interval==0 and step!=end:
                if shutil.disk_usage('/mnt/d').free < 25*1024**3:raise RuntimeError('insufficient host capacity for rolling recovery')
                recovery=run/'recovery';recovery.mkdir(exist_ok=True)
                cp_recovery=save(recovery,step,agent,replay,env,obs,rng,episode_id,episode_reward,severe,c,device,training_spec)
                # Verify both serialized headers and Replay hash before retiring a rolling package.
                check=torch.load(cp_recovery,map_location='cpu',weights_only=False)
                assert check['trainer']['step']==step and sha(recovery/check['replay_file'])==check['replay_sha256']
                pointer=recovery/'latest.json.tmp';pointer.write_text(json.dumps({'step':step,'checkpoint':str(cp_recovery),'sha256':sha(cp_recovery)}));os.replace(pointer,recovery/'latest.json')
                for old in sorted(recovery.glob('checkpoint_*.pt'))[:-2]:
                    old_payload=torch.load(old,map_location='cpu',weights_only=False)
                    old_replay=recovery/old_payload['replay_file']
                    assert old_replay.parent.resolve()==recovery.resolve() and old_replay.name.startswith('replay_')
                    old.unlink();old_replay.unlink()
                log(run/'events.jsonl',{'event':'verified_rolling_recovery','step':step,'file':str(cp_recovery)})
            if step in (20000,40000,80000,120000,160000,200000,240000) or step==end:
                
                if shutil.disk_usage('/mnt/d').free < 25*1024**3:raise RuntimeError('host D disk below 25 GiB safe pre-checkpoint threshold')
                t=time.perf_counter()
                if step in (40000,240000) or (end not in (20000,40000,80000,120000,160000,200000,240000) and step==end):
                    cp=save(run,step,agent,replay,env,obs,rng,episode_id,episode_reward,severe,c,device,training_spec)
                else:
                    cp=run/('checkpoint_%06d.pt'%step)
                    if cp.exists():raise FileExistsError('checkpoint already exists')
                    atomic_save(cp,{'format':'nf_staticlearnable_model_v1','contract':c,'training_spec':training_spec,'step':step,'actor':agent.actor.state_dict(),'critic':agent.critic.state_dict(),'actor_target':agent.actor_target.state_dict(),'critic_target':agent.critic_target.state_dict(),'fusion':'independent_global_learnable_logits','environment_course':capture(env).get('course')})
                with torch.no_grad():
                    ot=agent.tensor_observation(obs);diag={}
                    for owner in ('actor','critic'):
                        enc=getattr(agent,owner).encoder;_,details=enc(ot,True)
                        for branch in ('near','far'):
                            diag[owner+'_'+branch+'_rms']=float(details[branch].square().mean().sqrt())
                            diag[owner+'_'+branch+'_parameter_delta_l2']=float(sum((p.detach().cpu()-initial[owner][n]).square().sum().item() for n,p in getattr(agent,owner).named_parameters() if n.startswith('encoder.'+branch+'.'))**.5)
                    log(run/'checkpoint_features.jsonl',{'step':step,**diag})
                log(run/'events.jsonl',{'event':'checkpoint','step':step,'file':cp.name,'seconds':time.perf_counter()-t})
        log(run/'events.jsonl',{'event':'stopped','step':end,'complete':end==total})
    except BaseException as e:
        log(run/'events.jsonl',{'event':'failure','error':repr(e),'traceback':traceback.format_exc()})
        raise
    finally:env.close()

if __name__=='__main__':main()
