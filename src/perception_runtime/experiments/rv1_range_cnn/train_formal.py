"""Versioned, explicit-device rv1 trainer. Outputs are always a new run directory."""
import argparse, copy, hashlib, json, os, random, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config,digest
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from .lidar import LidarConfig
from .environment import make_course_env
from .td3 import RangeTD3
from .replay import SuccessReplay
from .provenance import SCHEMA,contract,sources,validate_contract
from .checkpoint import capture,restore_environment,rng_state,restore_rng,atomic_save

FORMAT='rv1_formal_v1'

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
    training_spec=digest({'seed':cfg['seed'],'total_steps':total,'diagnostics':cfg['diagnostics'],'td3':td,'pool_fingerprint':pool.fingerprint})
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
    rng=np.random.default_rng(cfg['seed']);agent=RangeTD3(td,c,device=device)
    env=make_course_env(pool);start=0;episode_id=0;episode_reward=0.;severe=0
    try:
        if args.resume:
            cp=args.resume.resolve()
            if cp.parent!=run:raise ValueError('resume checkpoint must be in run directory')
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            if payload.get('format')!=FORMAT or payload.get('schema')!=SCHEMA:raise ValueError('incompatible old checkpoint')
            validate_contract(payload['contract'],c)
            if payload.get('training_spec')!=training_spec:raise ValueError('training configuration or pool changed')
            if payload['device']!=str(device):raise ValueError('resume must use original device; device migration needs validation')
            name=payload['replay_file']
            if Path(name).name!=name:raise ValueError('invalid replay path')
            rp=run/name
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
            (run/'metadata.json').write_text(json.dumps({'format':FORMAT,'contract':c,'config':cfg,'source_hashes':sources(),'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda,'pool_fingerprint':pool.fingerprint},indent=2))
        log(run/'events.jsonl',{'event':'start' if start==0 else 'resume','step':start,'end':end,'device':str(device),'time':time.time()})
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
            if step%args.checkpoint_interval==0 or step==end:
                t=time.perf_counter();cp=save(run,step,agent,replay,env,obs,rng,episode_id,episode_reward,severe,c,device,training_spec)
                log(run/'events.jsonl',{'event':'checkpoint','step':step,'file':cp.name,'seconds':time.perf_counter()-t})
        log(run/'events.jsonl',{'event':'stopped','step':end,'complete':end==total})
    except BaseException as e:
        log(run/'events.jsonl',{'event':'failure','error':repr(e),'traceback':traceback.format_exc()})
        raise
    finally:env.close()

if __name__=='__main__':main()
