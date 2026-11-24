"""GPU-capable A2/A4 training with checkpoint/replay/physics recovery."""
import argparse,copy,hashlib,json,os,random,shutil,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.checkpoint import capture,restore_environment,rng_state,restore_rng,atomic_save
from ms_td3 import SCHEMA,ObservationEncoder,GeometricTD3,BalancedReplay

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(p,d):
    with p.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--mode',choices=['near','near_far'],required=True)
    p.add_argument('--stop-after',type=int,default=240000);p.add_argument('--interval',type=int,default=20000);p.add_argument('--resume',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--seed',type=int,default=43);a=p.parse_args()
    if not 1<=a.stop_after<=240000 or a.interval<1:raise ValueError('bounded budget')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    run=a.run_dir.resolve()
    if a.resume is None:run.mkdir(parents=True,exist_ok=False)
    elif not run.is_dir():raise FileNotFoundError(run)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    cfg=load_config();td=copy.deepcopy(cfg['td3']);seed=a.seed
    if a.preflight:td['warmup']=32;td['update_after']=32;td['batch_size']=32
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);action_rng=np.random.default_rng(seed)
    if a.device=='cuda':torch.cuda.manual_seed_all(seed)
    pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=True);encoder=ObservationEncoder(env.lidar.config);agent=GeometricTD3(td,device=a.device)
    base=Path(__file__).parent
    source_hash={name:sha(base/name) for name in ('ms_train_gpu.py','ms_td3.py','multiscale.py','fast_multiscale.py')}
    contract={'schema':SCHEMA,'mode':a.mode,'device':a.device,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None,'near_voxel_m':.4,'near_size':5,'grid':41,'max_range_m':8.,'state_dim':18,'feature_dim':302,'sensor':vars(env.lidar.config),'feature_order':'State18; near C xyz occ/free interleaved; far lex xyz occ/free interleaved','source_sha256':source_hash,'pool_fingerprint':pool.fingerprint,'td3':td,'seed':seed}
    def enc(raw):
        x=encoder(raw)
        if a.mode=='near':x[18+250:]=0
        return x
    def save(step,episode_id,episode_return,s,severe):
        target=run/('checkpoint_%06d.pt'%step)
        if target.exists():raise FileExistsError(target)
        host_free=shutil.disk_usage('/mnt/d').free
        if host_free<20*1024**3:raise OSError('D: has under 20 GiB free; checkpoint halted')
        data={'schema':SCHEMA,'contract':contract,'trainer':{'step':step,'episode_id':episode_id,'episode_return':episode_return,'severe':severe},'agent':agent.state(),'replay':replay,'environment':capture(env),'observation':s,'rng':rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if a.device=='cuda' else None,'action_rng':copy.deepcopy(action_rng.bit_generator.state)}
        atomic_save(target,data);log(run/'events.jsonl',{'event':'checkpoint','step':step,'file':target.name,'sha256':sha(target),'host_d_free_bytes_before':host_free})
    try:
        if a.resume:
            payload=torch.load(a.resume,map_location='cpu',weights_only=False)
            if payload['contract']!=contract:raise ValueError('source/config/representation mismatch')
            agent.restore(payload['agent']);replay=payload['replay'];raw=restore_environment(env,payload['environment']);s=enc(raw)
            np.testing.assert_array_equal(s,payload['observation']);restore_rng(payload['rng']);action_rng.bit_generator.state=payload['action_rng']
            if a.device=='cuda':
                if len(payload['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA topology changed')
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
            start=payload['trainer']['step'];episode_id=payload['trainer']['episode_id'];episode_return=payload['trainer']['episode_return'];severe=payload['trainer']['severe']
        else:
            replay=BalancedReplay(td['buffer_size'],td['success_replay_size'],td['success_replay_fraction'],seed);raw,_=env.reset(seed=seed);s=enc(raw)
            start=episode_id=severe=0;episode_return=0.;(run/'metadata.json').write_text(json.dumps({'contract':contract,'config':cfg,'note':'bounded diagnostic comparison, not formal success claim'},indent=2,default=str))
        if start>=a.stop_after:raise ValueError('stop must exceed resumed step')
        begin=time.perf_counter();log(run/'events.jsonl',{'event':'resume' if start else 'start','step':start,'end':a.stop_after})
        for step in range(start+1,a.stop_after+1):
            if step<=td['warmup']:action=action_rng.uniform(-1,1,3)
            else:
                sigma=max(td['exploration_end'],td['exploration_start']-(td['exploration_start']-td['exploration_end'])*step/td['noise_decay_steps'])
                action=agent.action(s)+action_rng.normal(0,sigma,3)
            action=project(torch.as_tensor(action,dtype=torch.float32)).numpy();before=env.lidar.cache.timestamp_s
            raw,reward,terminated,truncated,info=env.step(action)
            if env.lidar.cache.timestamp_s<=before or env.lidar.scan_count!=env.upper_tick+1:raise RuntimeError('scan timing/count')
            nxt=enc(raw);executed=np.asarray(info['projected_action'],np.float32);replay.add(s,executed,nxt,reward*cfg['reward_scale'],terminated,truncated);s=nxt;episode_return+=reward
            if step>=td['update_after']:
                d=agent.update(replay);bad=d['q_abs_max']>cfg['diagnostics']['severe_q_abs'] or d['critic_loss']>cfg['diagnostics']['severe_loss'];severe=severe+1 if bad else 0
                log(run/'updates.jsonl',{'step':step,**d})
                if agent.updates in (1,2):
                    devices={name:str(next(getattr(agent,name).parameters()).device) for name in ('actor','critic','actor_target','critic_target')}
                    if any(not value.startswith(a.device) for value in devices.values()):raise RuntimeError('network device mismatch')
                    log(run/'events.jsonl',{'event':'device_check','step':step,'devices':devices,'actor_updated':d['actor_updated']})
                if severe>=cfg['diagnostics']['severe_consecutive_updates']:raise FloatingPointError('severe learning condition')
            if terminated or truncated:
                reason=info['done_reason'];replay.finish(reason=='success',(10,20,30,40)[info['density_stage']]);env.record_episode_outcome(reason=='success',collision=info['collision'],done_reason=reason)
                log(run/'episodes.jsonl',{'step':step,'episode':episode_id,'reward':episode_return,'done_reason':reason,'density':int(info['density_stage'])})
                episode_id+=1;episode_return=0.;raw,_=env.reset();s=enc(raw)
            if step%500==0:
                progress={'step':step,'updates':agent.updates,'episodes':episode_id,'steps_per_second':(step-start)/(time.perf_counter()-begin),'success_sizes':{str(k):v.size for k,v in replay.success.items()}}
                log(run/'metrics.jsonl',progress);print(json.dumps(progress),flush=True)
            if step%a.interval==0 or step==a.stop_after:save(step,episode_id,episode_return,s,severe)
        log(run/'events.jsonl',{'event':'stopped','step':step})
    except BaseException:
        log(run/'events.jsonl',{'event':'failure','step':locals().get('step',0),'traceback':traceback.format_exc()});raise
    finally:env.close()

if __name__=='__main__':main()
