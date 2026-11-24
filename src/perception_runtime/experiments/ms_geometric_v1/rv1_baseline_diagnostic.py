"""Bounded frozen RV1 diagnostic matching the MSGP diagnostic episode protocol."""
import argparse,json,time
from pathlib import Path
import torch
from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.td3 import RangeTD3

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cfg=load_config();payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
agent=RangeTD3(cfg['td3'],payload['contract']);agent.restore(payload['agent']);pool=ForestPool(ROOT/cfg['forest_pool_path']);env=make_course_env(pool,training=False)
records=[];episodes=[];t0=time.perf_counter()
try:
 for ep in range(6):
  obs,info=env.reset(seed=100+ep);reason='running';n=0;zmin=1e6;zmax=-1e6
  while n<320 and reason=='running':
   action=agent.action(obs);z=float(env.base_env.pos[0,2]);zmin=min(zmin,z);zmax=max(zmax,z)
   obs,reward,term,trunc,info=env.step(action);reason=info['done_reason'];n+=1
   records.append({'episode':ep,'step':n,'z':z,'vz_cmd_norm':float(action[2]),'goal_distance':float(info['goal_distance']),'reason':reason})
  episodes.append({'episode':ep,'steps':n,'reason':reason,'z_min':zmin,'z_max':zmax,'seed':100+ep});print(json.dumps(episodes[-1]),flush=True)
 (out/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
 summary={'checkpoint_step':payload['trainer']['step'],'split':'validation','deterministic':True,'episodes':episodes,'elapsed_s':time.perf_counter()-t0,'scope':'diagnostic only; six unverified-pairing episodes, not formal evaluation'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
finally:env.close()
