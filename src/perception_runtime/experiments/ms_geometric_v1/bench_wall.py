import time,json,numpy as np,pybullet as p
from experiments.rv1_range_cnn.lidar import Lidar,LidarConfig
from multiscale import rv1_directions
from fast_multiscale import FastEncoder
client=p.connect(p.DIRECT); c=LidarConfig(); sensor=Lidar(c); e=FastEncoder(); d=rv1_directions()
try:
 wall=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.1,3,2],physicsClientId=client);p.createMultiBody(baseMass=0,baseCollisionShapeIndex=wall,basePosition=[2,0,1.02],physicsClientId=client)
 times=[];scan_t=[];voxel_t=[];feature_t=[]
 for i in range(100):
  t0=time.perf_counter();scan=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=i/15);t1=time.perf_counter()
  g=e.voxelize(d,scan.range_m,scan.hit_mask,scan.scanned_mask);t2=time.perf_counter(); f=e.features(g);t3=time.perf_counter()
  scan_t.append((t1-t0)*1000);voxel_t.append((t2-t1)*1000);feature_t.append((t3-t2)*1000);times.append((t3-t0)*1000)
 assert 1.9 < scan.range_m[15,0] < 1.91
 assert g[25,20,20]==2 and g[27,20,20]==0,(g[25,20,20],g[27,20,20])
 assert g[20,20,30]==0
 def stats(x):return {'p50_ms':float(np.percentile(x,50)),'p95_ms':float(np.percentile(x,95)),'max_ms':float(max(x))}
 print(json.dumps(dict(frames=100,scene='wall',checks=['wall hit range','occupied wall','occluded behind wall unknown','outside vertical fov unknown'],scan=stats(scan_t),voxel=stats(voxel_t),features=stats(feature_t),scan_to_features=stats(times)),indent=2))
finally:p.disconnect(client)
