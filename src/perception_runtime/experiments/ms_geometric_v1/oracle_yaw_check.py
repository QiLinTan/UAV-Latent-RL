import json,time,math,numpy as np,pybullet as p
from multiscale import Encoder,rv1_directions,REGIONS
from fast_multiscale import FastEncoder
from experiments.rv1_range_cnn.lidar import Lidar,LidarConfig
c=Encoder().c;fast=FastEncoder();ref=Encoder(); centers=(np.arange(c.n)-c.n//2)*c.voxel
xyz=np.stack(np.meshgrid(centers,centers,centers,indexing='ij'),axis=-1); low=xyz-c.voxel/2; high=xyz+c.voxel/2; origin=np.array([0.,0.,c.origin_height])
def oracle(d,r,h,s):
 free=np.zeros((c.n,)*3,bool);occ=np.zeros_like(free)
 for v,dist,hit,scanned in zip(d,r,h,s):
  if not scanned:continue
  tmin=np.full(free.shape,-np.inf);tmax=np.full(free.shape,np.inf)
  for axis in range(3):
   if abs(v[axis])<1e-14:
    allowed=(low[...,axis] <= origin[axis]) & (origin[axis]< high[...,axis]);tmin=np.where(allowed,tmin,np.inf)
   else:
    x=(low[...,axis]-origin[axis])/v[axis];y=(high[...,axis]-origin[axis])/v[axis];tmin=np.maximum(tmin,np.minimum(x,y));tmax=np.minimum(tmax,np.maximum(x,y))
  crossed=(np.minimum(tmax,dist)>np.maximum(tmin,c.start_range)+1e-11)
  end=np.floor((origin+v*dist+fast.half)/c.voxel).astype(int)
  if hit and np.all((end>=0)&(end<c.n)): crossed[tuple(end)]=False;occ[tuple(end)]=True
  free|=crossed
 return np.where(occ,2,np.where(free,1,0)).astype(np.uint8)
rng=np.random.default_rng(20261118); results=[]
for n in [1,16,48]:
 d=rng.normal(size=(n,3));d/=np.linalg.norm(d,axis=1)[:,None];s=rng.random(n)>.15;h=(rng.random(n)>.35)&s;r=np.where(h,rng.uniform(.09,7.99,n),8.);r[~s]=np.nan
 o=oracle(d,r,h,s);a=ref.voxelize(d,r,h,s);b=fast.voxelize(d,r,h,s);results.append(dict(rays=n,ref_diff=int(np.count_nonzero(o!=a)),fast_diff=int(np.count_nonzero(o!=b))))
 assert np.array_equal(o,a) and np.array_equal(o,b),results
client=p.connect(p.DIRECT);cfg=LidarConfig(); sensor=Lidar(cfg)
try:
 shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.1,3.,2.],physicsClientId=client);p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[2,0,1.02],physicsClientId=client)
 scans=[]
 for yaw in [0.,math.pi/2]:
  q=p.getQuaternionFromEuler([0,0,yaw]); sc=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=q,timestamp_s=0)
  g=fast.voxelize(rv1_directions(),sc.range_m,sc.hit_mask,sc.scanned_mask);f=fast.features(g)[250:].reshape(26,2)
  scans.append(dict(yaw=yaw,hits=int(sc.hit_mask.sum()),plus_x=float(f[REGIONS.index((2,1,1)),0]),minus_y=float(f[REGIONS.index((1,0,1)),0]),range_forward=float(sc.range_m[15,0]),range_right=float(sc.range_m[15,96])))
 assert scans[0]['plus_x']>0 and scans[0]['range_forward']<2
 assert scans[1]['minus_y']>0 and scans[1]['range_right']<2
 assert scans[1]['range_forward']>7.99
finally:p.disconnect(client)
print(json.dumps(dict(oracle=results,yaw=scans,collision_urdf={'type':'cylinder','radius_m':.06,'length_m':.025,'note':'URDF collision geometry only; safety margin not included'}),indent=2))
