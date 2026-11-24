import json,time
import numpy as np
from multiscale import Encoder,rv1_directions,REGIONS
enc=Encoder(); d=rv1_directions(); dx=d[...,0]; scanned=np.ones(dx.shape,bool)
records=[]
for wall in [2.,4.,6.,None]:
    dist=np.full(dx.shape,8.); hit=np.zeros(dx.shape,bool)
    if wall:
        t=np.divide(wall,dx,out=np.full_like(dx,np.inf),where=dx>0)
        hit=(t<=8);dist[hit]=t[hit]
    times=[]
    for _ in range(4):
        t0=time.perf_counter(); grid=enc.voxelize(d,dist,hit,scanned); t1=time.perf_counter(); f=enc.features(grid); t2=time.perf_counter()
        times.append([1000*(t1-t0),1000*(t2-t1)])
    near=f[:250].reshape(-1,2);far=f[250:].reshape(26,2)
    records.append(dict(wall_x_m=wall, hit_rays=int(hit.sum()), grid_counts=np.bincount(grid.ravel(),minlength=3).tolist(), near_occ=int(near[:,0].sum()),far_max_occ=float(far[:,0].max()),far_sum_occ=float(far[:,0].sum()), far_max_free=float(far[:,1].max()), timings_ms=times, positive_x_center=far[REGIONS.index((2,1,1))].tolist()))
print(json.dumps(records,indent=2))
