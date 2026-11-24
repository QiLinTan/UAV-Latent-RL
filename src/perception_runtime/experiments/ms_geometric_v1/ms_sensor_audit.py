"""Diagnostic only: sample actual scan type and range boundaries in training environment."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.environment import make_course_env

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=load_config();pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=True);rng=np.random.default_rng(20261118)
counts=np.zeros(5,np.int64);scan_count=0;at_start=0;near_start=0;min_dist=8.;outcomes={};t0=time.perf_counter();rows=[]
try:
 obs,info=env.reset(seed=20261118)
 for step in range(501):
  scan=env.lidar.cache;valid=scan.scanned_mask;counts+=np.bincount(scan.hit_type.ravel(),minlength=5)[:5]
  values=scan.range_m[valid];min_dist=min(min_dist,float(values.min()));at_start+=int((scan.hit_mask & (scan.range_m<=np.float32(scan.config.origin_offset_m))).sum());near_start+=int((scan.hit_mask & (scan.range_m<=.1)).sum());scan_count+=1
  rows.append({'step':step,'self_hits':int((scan.hit_type==1).sum()),'ground_hits':int((scan.hit_type==2).sum()),'obstacle_hits':int((scan.hit_type==3).sum()),'other_hits':int((scan.hit_type==4).sum()),'min_range':float(values.min())})
  if step==500:break
  action=project(torch.tensor(rng.uniform(-1,1,3),dtype=torch.float32)).numpy();obs,r,t,u,info=env.step(action)
  if t or u:
   reason=info['done_reason'];outcomes[reason]=outcomes.get(reason,0)+1;env.record_episode_outcome(reason=='success',collision=info['collision'],done_reason=reason);obs,info=env.reset()
 summary={'scans':scan_count,'rays_per_scan':env.lidar.config.height*env.lidar.config.width,'hit_type_counts_none_self_ground_obstacle_other':counts.tolist(),'at_start_hits':at_start,'hits_le_0.1m':near_start,'minimum_m':min_dist,'outcomes':outcomes,'elapsed_s':time.perf_counter()-t0,'scope':'500 random actions in training split; not exhaustive'}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'scans.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps(summary,indent=2))
finally:env.close()
