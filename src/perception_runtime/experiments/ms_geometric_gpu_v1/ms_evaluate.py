"""Fixed Stage0/3 validation pool, original starts, deterministic policy."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.td3 import RangeTD3
from ms_td3 import GeometricTD3,ObservationEncoder,SCHEMA

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['a1','a2','a4'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=20);a=p.parse_args()
 if not 1<=a.count<=20:raise ValueError('count')
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cfg=load_config();data=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if a.kind=='a1':
  agent=RangeTD3(cfg['td3'],data['contract']);agent.restore(data['agent']);step=data['trainer']['step']
 else:
  if data['schema']!=SCHEMA or data['contract']['mode']!=('near' if a.kind=='a2' else 'near_far'):raise ValueError('incompatible checkpoint')
  agent=GeometricTD3(cfg['td3']);agent.restore(data['agent']);step=data['trainer']['step']
 pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=False);encoder=ObservationEncoder(env.lidar.config)
 protocol={'kind':a.kind,'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'split':'validation','stages':[0,3],'seeds':list(range(a.count)),'start':'original level4','deterministic':True,'limit_upper_steps':2000,'pool_fingerprint':pool.fingerprint,'note':'preliminary single-seed comparison; not held-out test'}
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
 try:
  for stage in [0,3]:
   env.obstacle_curriculum.stage=stage
   for seed in range(a.count):
    obs,info=env.reset(seed=seed);b=env.base_env
    if info['density_stage']!=stage or info['start_curriculum_source']!='original':raise RuntimeError('wrong validation setup')
    init={'position':b.pos[0].tolist(),'quaternion':b.quat[0].tolist(),'physical_state20':np.asarray(b._getDroneStateVector(0)).tolist(),'target':np.asarray(b.TARGET_POS).tolist(),'geometry_sha256':info['geometry_sha256'],'map_pool_index':info['map_pool_index']}
    prior=np.asarray(b.pos[0]).copy();path=0.;clear=[];start=time.perf_counter();reason='running';n=0;lower_ticks=0
    while reason=='running' and n<2000:
     if a.kind=='a1':act=agent.action(obs)
     else:
      x=encoder(obs)
      if a.kind=='a2':x[268:]=0
      act=agent.action(x)
     obs,reward,t,u,info=env.step(act);n+=1;reason=info['done_reason']
     for tick in env.last_ticks:
      cur=np.asarray(tick['position']);path+=float(np.linalg.norm(cur-prior));prior=cur.copy();lower_ticks+=1
     value=float(info['min_tree_clearance'])
     if np.isfinite(value):clear.append(value)
    if reason=='running':raise RuntimeError('unexpected evaluation horizon')
    result={'stage':stage,'seed':seed,'initial':init,'reason':reason,'success':reason=='success','upper_steps':n,'lower_ticks':lower_ticks,'duration_s':lower_ticks/120.,'path_m':path,'min_tree_clearance_as_env_m':min(clear) if clear else None,'final_goal_distance_m':float(info['goal_distance']),'wall_s':time.perf_counter()-start}
    results.append(result);(a.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
  (a.output/'summary.json').write_text(json.dumps({'protocol':protocol,'results_count':len(results),'success':sum(r['success'] for r in results),'counts_by_stage':{str(s):{reason:sum(r['stage']==s and r['reason']==reason for r in results) for reason in sorted({r['reason'] for r in results if r['stage']==s})} for s in [0,3]}},indent=2))
 finally:env.close()
if __name__=='__main__':main()
