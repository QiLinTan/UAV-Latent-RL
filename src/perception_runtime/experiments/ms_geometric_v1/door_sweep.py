import json,numpy as np,pybullet as p
from experiments.rv1_range_cnn.lidar import Lidar,LidarConfig
from multiscale import rv1_directions
from fast_multiscale import FastEncoder
client=p.connect(p.DIRECT); enc=FastEncoder(); records=[]; features={}
try:
 for x in [.8,1.2,2.0]:
  for gap in [.2,.4,.8]:
   p.resetSimulation(physicsClientId=client);sensor=Lidar(LidarConfig())
   for sign in [-1,1]:
    half=[.05,(3-gap/2)/2,1.5];y=sign*(gap/2+half[1]); shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client);p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[x,y,1.02],physicsClientId=client)
   sc=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0)
   f,g=enc.encode(rv1_directions(),sc.range_m,sc.hit_mask,sc.scanned_mask);features[(x,gap)]=f.copy();near=f[:250].reshape(5,5,5,2)
   records.append(dict(wall_x=x,gap_m=gap,hit_rays=int(sc.hit_mask.sum()),forward_ray_hit=bool(sc.hit_mask[15,0]),near_occ_total=int(near[:,:,:,0].sum()),near_occ_midline=int(near[:,2,:,0].sum()),near_free_midline=int(near[:,2,:,1].sum()),far_max_occ=float(f[250:].reshape(26,2)[:,0].max())))
 pairs=[dict(wall_x=x,near_different=int(np.count_nonzero(features[(x,.2)][:250]!=features[(x,.4)][:250])),far_different=int(np.count_nonzero(features[(x,.2)][250:]!=features[(x,.4)][250:]))) for x in [.8,1.2,2.0]]
 print(json.dumps(dict(comparison_02_vs_04=pairs,collision_diameter_m=.12,qualification='collision cylinder only; no margin or propeller envelope',cases=records),indent=2))
finally:p.disconnect(client)

