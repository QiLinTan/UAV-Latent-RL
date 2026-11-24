import json,time, numpy as np, pybullet as p
from experiments.rv1_range_cnn.lidar import Lidar,LidarConfig
from multiscale import Encoder,rv1_directions
from fast_multiscale import FastEncoder
c=LidarConfig(); ref=Encoder(); fast=FastEncoder(); d=rv1_directions(); rng=np.random.default_rng(43)
for i in range(25):
 ds=rng.normal(size=(200,3)); ds/=np.linalg.norm(ds,axis=1)[:,None]; scanned=rng.random(200)>.12; hit=(rng.random(200)>.3)&scanned
 ranges=np.where(hit,rng.uniform(.08,8,200),8.); ranges[~scanned]=np.nan
 a=ref.voxelize(ds,ranges,hit,scanned); b=fast.voxelize(ds,ranges,hit,scanned)
 if not np.array_equal(a,b): raise AssertionError(f'random mismatch {i}: {np.count_nonzero(a!=b)}')
for ds,rs,hs in [(np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0],[1,1,0]/np.sqrt(2)]),np.array([.2,1.,8,2.,4.]),np.array([1,1,0,1,1],bool))]:
 assert np.array_equal(ref.voxelize(ds,rs,hs,np.ones(len(rs),bool)),fast.voxelize(ds,rs,hs,np.ones(len(rs),bool)))
client=p.connect(p.DIRECT); p.setGravity(0,0,0,physicsClientId=client); sensor=Lidar(c); records=[]
try:
 for label,boxes in [('empty',[]),('wall',[( [2.,0.,1.02],[.1,3.,2.])]),('rod',[( [2.,0.,1.02],[.08,.08,1.])]),('door',[( [2.,-1.25,1.02],[.1,.75,2.]),([2.,1.25,1.02],[.1,.75,2.])])]:
  p.resetSimulation(physicsClientId=client)
  for pos,half in boxes:
   shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client); p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=pos,physicsClientId=client)
  t=time.perf_counter(); scan=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0); scan_ms=(time.perf_counter()-t)*1000
  a=ref.voxelize(d,scan.range_m,scan.hit_mask,scan.scanned_mask); b=fast.voxelize(d,scan.range_m,scan.hit_mask,scan.scanned_mask)
  assert np.array_equal(a,b),label
  if label=='empty': assert not scan.hit_mask.any()
  else: assert scan.hit_mask.any()
  np.savez_compressed('scene_'+label+'.npz',range_m=scan.range_m,hit_mask=scan.hit_mask,scanned_mask=scan.scanned_mask,grid=a,features=fast.features(a))
  times=[]
  for i in range(30):
   t=time.perf_counter(); fast.voxelize(d,scan.range_m,scan.hit_mask,scan.scanned_mask); times.append((time.perf_counter()-t)*1000)
  f=fast.features(a)[250:].reshape(26,2)
  records.append(dict(scene=label,hits=int(scan.hit_mask.sum()),scan_single_ms=scan_ms,fast_voxel_p50_ms=float(np.percentile(times,50)),fast_voxel_p95_ms=float(np.percentile(times,95)),far_occ_max=float(f[:,0].max())))
 print(json.dumps(dict(random_cases=25,random_rays_per_case=200,axis_case=True,scenes=records),indent=2))
finally:p.disconnect(client)
