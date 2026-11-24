import json,time,numpy as np,pybullet as p
from experiments.rv1_range_cnn.lidar import Lidar,LidarConfig
from multiscale import rv1_directions,Config
from fast_multiscale import FastEncoder
client=p.connect(p.DIRECT); c04=FastEncoder(); c02=FastEncoder(Config(voxel=.2,n=81,near=9)); result=[]
try:
 for y_shift in [-.1,0.,.1]:
  frames={}
  for gap in [.2,.4]:
   p.resetSimulation(physicsClientId=client)
   for sign in [-1,1]:
    half=[.05,(3-gap/2)/2,1.5];y=y_shift+sign*(gap/2+half[1]);shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client);p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[.8,y,1.02],physicsClientId=client)
   scan=Lidar(LidarConfig()).scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0)
   rows={}
   for name,enc in [('0.4m',c04),('0.2m',c02)]:
    t=time.perf_counter();f,g=enc.encode(rv1_directions(),scan.range_m,scan.hit_mask,scan.scanned_mask);ms=(time.perf_counter()-t)*1000;rows[name]=f;frames[(gap,name)]=f
    result.append(dict(offset_y=y_shift,gap=gap,voxel=name,near_dims=2*enc.c.near**3,near_occupied=int(f[:2*enc.c.near**3].reshape(-1,2)[:,0].sum()),encoding_single_ms=ms))
  for name,enc in [('0.4m',c04),('0.2m',c02)]:
   n=2*enc.c.near**3;result.append(dict(offset_y=y_shift,voxel=name,comparison='0.2 vs 0.4 gap',near_changed_dimensions=int(np.count_nonzero(frames[(.2,name)][:n]!=frames[(.4,name)][:n]))))
 print(json.dumps(dict(note='near half widths: 1.0m for 0.4 grid; 0.9m for 0.2 grid; single timings diagnostic only',cases=result),indent=2))
finally:p.disconnect(client)
