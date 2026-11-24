"""Bounded, supervised evaluation of three frozen seed43 policies at seven nodes."""
import concurrent.futures,collections,json,os,pathlib,subprocess,sys,time,traceback

ROOT=pathlib.Path('/opt/uav-training-20261103/rv1_project')
RUNS=pathlib.Path('/opt/uav-training-20261103/runs')
SRC=ROOT/'experiments/ms_geometric_gpu_v1/ms_evaluate_nodes.py'
OUT=RUNS/'formal_gpu_seed43_seven_nodes_eval_20261118a'
PY='/opt/uav-training-20261103/venv/bin/python'
NODES=(20000,40000,80000,120000,160000,200000,240000)
TRAIN={'a1':RUNS/'formal_gpu_a1_seed43_240k_20261118a','a2':RUNS/'formal_gpu_a2_seed43_240k_20261118a','a4':RUNS/'formal_gpu_a4_seed43_240k_20261118a'}

def main():
 expected={}
 for kind,run in TRAIN.items():
  lines=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
  if not any(x.get('event')=='stopped' and x.get('step')==240000 for x in lines):raise RuntimeError('training not complete '+kind)
  expected[kind]={x['step']:x.get('sha256') for x in lines if x.get('event')=='checkpoint'}
  for step in NODES:
   if not (run/f'checkpoint_{step:06d}.pt').is_file():raise FileNotFoundError((kind,step))
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'logs').mkdir()
 jobs=[(kind,step) for step in NODES for kind in ('a1','a2','a4')];started=time.time()
 config={'nodes':NODES,'groups':list(TRAIN),'validation_stages':[0,3],'seeds':list(range(20)),'training_seed':43,'workers':6,'torch_threads_per_worker':2,'inference_device':'cpu','origin':'fixed validation pool; original start; not sealed test','case_count':40}
 (OUT/'queue_config.json').write_text(json.dumps(config,indent=2))
 def execute(job):
  kind,step=job;name=f'{kind}_{step:06d}';t=time.time()
  cp=TRAIN[kind]/f'checkpoint_{step:06d}.pt';dest=OUT/name;logfile=OUT/'logs'/f'{name}.log'
  args=[PY,'-u',str(SRC),'--kind',kind,'--checkpoint',str(cp),'--output',str(dest)]
  env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+':'+str(SRC.parent);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
  with logfile.open('w') as stream:rc=subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
  if rc:return {'job':name,'status':'FAILED','exit_code':rc,'log':str(logfile),'elapsed_s':time.time()-t}
  summary=json.loads((dest/'summary.json').read_text());rows=json.loads((dest/'results.json').read_text());proto=summary['protocol']
  if len(rows)!=40 or summary['results_count']!=40 or proto['checkpoint_step']!=step or proto['kind']!=kind:raise ValueError('unexpected evaluation output '+name)
  if expected[kind].get(step) and expected[kind][step]!=proto['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch '+name)
  return {'job':name,'status':'PASS','elapsed_s':time.time()-t,'output':str(dest)}
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures={pool.submit(execute,j):j for j in jobs}
  for future in concurrent.futures.as_completed(futures):
   job=futures[future]
   try:r=future.result()
   except Exception:r={'job':f'{job[0]}_{job[1]:06d}','status':'FAILED','traceback':traceback.format_exc()}
   results.append(r);(OUT/'queue_status.json').write_text(json.dumps({'completed':len(results),'total':len(jobs),'jobs':results},indent=2));print(json.dumps({'completed':len(results),'total':len(jobs),**r}),flush=True)
 if any(r['status']!='PASS' for r in results):raise RuntimeError('one or more evaluation jobs failed; inspect queue_status')
 baseline=None;report=[]
 for step in NODES:
  per={}
  for kind in ('a1','a2','a4'):
   root=OUT/f'{kind}_{step:06d}';rows=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
   key=[(r['stage'],r['seed'],r['initial']) for r in rows]
   if baseline is None:baseline=key
   if key!=baseline:raise RuntimeError('unpaired scenario/pose '+kind+' '+str(step))
   stages={}
   for stage in (0,3):
    part=[r for r in rows if r['stage']==stage];counts=dict(collections.Counter(r['reason'] for r in part))
    stages[str(stage)]={'n':len(part),'outcomes':counts,'success':sum(r['success'] for r in part),'success_without_observed_collision':sum(r['success_without_observed_collision'] for r in part),'collision_observed':sum(r['collision_observed'] for r in part)}
   per[kind]={'stages':stages,'checkpoint_sha256':protocol['checkpoint_sha256']}
  paired={}
  r4=json.loads((OUT/f'a4_{step:06d}'/'results.json').read_text());r2=json.loads((OUT/f'a2_{step:06d}'/'results.json').read_text())
  for stage in (0,3):
   pair=[(x,y) for x,y in zip(r4,r2) if x['stage']==stage]
   paired[str(stage)]={'both_success':sum(x['success'] and y['success'] for x,y in pair),'a4_only_success':sum(x['success'] and not y['success'] for x,y in pair),'a2_only_success':sum(not x['success'] and y['success'] for x,y in pair),'both_fail':sum(not x['success'] and not y['success'] for x,y in pair)}
  report.append({'step':step,'models':per,'paired_a4_vs_a2':paired})
 (OUT/'comparison.json').write_text(json.dumps({'config':config,'all_840_cases_initial_geometry_matched':True,'nodes':report,'elapsed_s':time.time()-started},indent=2));print(json.dumps({'event':'complete','path':str(OUT/'comparison.json'),'elapsed_s':time.time()-started}),flush=True)

if __name__=='__main__':main()
