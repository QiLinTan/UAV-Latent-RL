"""Real RV1 observations; exercise all A1/A2/A4 GPU TD3 updates at batch 256."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,random,time
import numpy as np,torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.provenance import contract
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.replay import SuccessReplay
from ms_td3 import GeometricTD3,BalancedReplay,ObservationEncoder

torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
cfg=load_config();td=cfg['td3'];pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=True);enc=ObservationEncoder(env.lidar.config)
c=contract(LidarConfig(),'gpu_batch_preflight');rng=np.random.default_rng(43)
replays={'a1':SuccessReplay(c,capacity=64,success_capacity=8,fraction=.2),'a2':BalancedReplay(64,8,.2),'a4':BalancedReplay(64,8,.2)}
try:
 obs,_=env.reset(seed=43)
 for i in range(64):
  before=env.lidar.cache.timestamp_s;action=project(torch.tensor(rng.uniform(-1,1,3),dtype=torch.float32)).numpy();n,reward,t,u,info=env.step(action);after=env.lidar.cache.timestamp_s
  s4=enc(obs);n4=enc(n);s2=s4.copy();n2=n4.copy();s2[268:]=0;n2[268:]=0
  aa=np.asarray(info['projected_action'],np.float32);replays['a1'].add(obs,aa,n,reward*cfg['reward_scale'],t,u,episode_id=0,timestamps=(before,after))
  replays['a2'].add(s2,aa,n2,reward*cfg['reward_scale'],t,u);replays['a4'].add(s4,aa,n4,reward*cfg['reward_scale'],t,u);obs=n
  if t or u:
   for r in replays.values():r.finish(info['done_reason']=='success',(10,20,30,40)[info['density_stage']])
   env.record_episode_outcome(info['done_reason']=='success',collision=info['collision'],done_reason=info['done_reason']);obs,_=env.reset()
finally:env.close()

results={}
for kind in ('a1','a2','a4'):
 torch.manual_seed(43);torch.cuda.manual_seed_all(43);np.random.seed(43);random.seed(43);torch.cuda.reset_peak_memory_stats()
 agent=RangeTD3(td,c,device='cuda') if kind=='a1' else GeometricTD3(td,device='cuda');replay=replays[kind]
 start=time.perf_counter();d1=agent.update(replay);d2=agent.update(replay);torch.cuda.synchronize();seconds=time.perf_counter()-start
 if not d2['actor_updated'] or not np.isfinite(d1['critic_loss']) or not np.isfinite(d2['actor_loss']):raise RuntimeError('update failed')
 for name in ('actor','critic','actor_target','critic_target'):
  if next(getattr(agent,name).parameters()).device.type!='cuda':raise RuntimeError('wrong device')
 if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.actor.parameters()):raise RuntimeError('actor gradient')
 if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.critic.parameters()):raise RuntimeError('critic gradient')
 clone=RangeTD3(td,c,device='cuda') if kind=='a1' else GeometricTD3(td,device='cuda');clone.restore(copy.deepcopy(agent.state()));r2=copy.deepcopy(replay)
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);x=agent.update(replay)
 torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);y=clone.update(r2)
 if x!=y or any(not torch.equal(a,b) for a,b in zip(agent.critic.parameters(),clone.critic.parameters())):raise RuntimeError('CUDA resume next-update mismatch '+kind)
 results[kind]={'batch':td['batch_size'],'device':'cuda','first_two_updates_s':seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'critic_loss':d2['critic_loss'],'actor_loss':d2['actor_loss'],'gradients_finite':True,'restored_next_update_exact':True}
print(json.dumps(results,indent=2))
