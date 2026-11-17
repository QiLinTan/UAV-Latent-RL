"""Batch-vectorized DDA; compare against multiscale.Encoder before use."""
import numpy as np
from multiscale import Encoder
class FastEncoder(Encoder):
 def voxelize(self,directions,ranges,hit,scanned):
  c=self.c; d=np.asarray(directions,float).reshape(-1,3); r=np.asarray(ranges,float).ravel(); h=np.asarray(hit,bool).ravel(); s=np.asarray(scanned,bool).ravel()
  if not(len(d)==len(r)==len(h)==len(s)): raise ValueError('shape mismatch')
  if np.any(h & ~s): raise ValueError('hit on unscanned ray')
  if np.any(s & ~h & (np.abs(r-c.max_range)>1e-5)): raise ValueError('scanned no-hit must be max range')
  if np.any(~np.isfinite(d)) or np.any(np.abs(np.linalg.norm(d,axis=1)-1)>1e-5): raise ValueError('unit directions required')
  if np.any(s & (~np.isfinite(r)|(r<c.start_range)|(r>c.max_range+1e-5))): raise ValueError('invalid scanned range')
  d=d[s]; r=np.minimum(r[s],c.max_range); h=h[s]; origin=np.array([0.,0.,c.origin_height]); eps=1e-10
  cell=np.floor((origin+d*(c.start_range+eps)+self.half)/c.voxel).astype(int); step=np.sign(d).astype(int)
  boundary=-self.half+(cell+(step>0))*c.voxel
  tmax=np.full(d.shape,np.inf); td=np.full(d.shape,np.inf); nz=step!=0
  np.divide(boundary-origin,d,out=tmax,where=nz); np.divide(c.voxel,np.abs(d),out=td,where=nz)
  end=np.floor((origin+d*r[:,None]+self.half)/c.voxel).astype(int)
  grid=np.zeros((c.n,)*3,np.uint8); active=np.ones(len(r),bool)
  while active.any():
   active &= np.all((cell>=0)&(cell<c.n),axis=1)
   active &= ~(h & np.all(cell==end,axis=1))
   ids=np.flatnonzero(active)
   if not len(ids): break
   xyz=cell[ids]; grid[xyz[:,0],xyz[:,1],xyz[:,2]]=1
   nt=tmax[ids].min(axis=1); done=nt>=r[ids]-eps; active[ids[done]]=False
   run=ids[~done]; axes=np.abs(tmax[run]-nt[~done,None])<eps
   cell[run]+=step[run]*axes; tmax[run]=np.where(axes,tmax[run]+td[run],tmax[run])
  ends=end[h]; ok=np.all((ends>=0)&(ends<c.n),axis=1); ends=ends[ok]
  grid[ends[:,0],ends[:,1],ends[:,2]]=2
  return grid
