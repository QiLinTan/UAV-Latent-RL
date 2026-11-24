"""Experimental single-frame RV1 multi-scale encoder. Not yet flight-validated."""
from dataclasses import dataclass
from itertools import product
import numpy as np

REGIONS = tuple(p for p in product(range(3), repeat=3) if p != (1, 1, 1))

def paper53(grid):
    a = np.asarray(grid)
    if a.shape != (9,9,9) or not np.isin(a,[0,1]).all():
        raise ValueError('expected binary 9x9x9 grid, axes x,y,z')
    near = a[3:6,3:6,3:6].ravel(order='C')
    far = [a[x*3:(x+1)*3,y*3:(y+1)*3,z*3:(z+1)*3].mean() for x,y,z in REGIONS]
    return np.concatenate((near,far)).astype(np.float32)

@dataclass(frozen=True)
class Config:
    voxel: float = .4
    n: int = 41
    near: int = 5
    max_range: float = 8.
    origin_height: float = .02
    start_range: float = .08

class Encoder:
    def __init__(self, config=Config()):
        self.c = c = config
        if c.voxel<=0 or c.n%2!=1 or c.near%2!=1 or c.near>=c.n or c.max_range<=c.start_range or c.start_range<0:
            raise ValueError('invalid geometry')
        self.half=c.n*c.voxel/2
        # A voxel is valid if its closed box intersects the range sphere around lidar origin.
        centers=(np.arange(c.n)-c.n//2)*c.voxel
        xyz=np.stack(np.meshgrid(centers,centers,centers,indexing='ij'),axis=-1)
        delta=np.abs(xyz-np.array([0,0,c.origin_height]))-c.voxel/2
        self.valid=(np.maximum(delta,0)**2).sum(axis=-1)<=c.max_range**2
        idx=np.indices((c.n,)*3); lo=(c.n-c.near)//2; hi=lo+c.near
        bands=np.where(idx<lo,0,np.where(idx<hi,1,2))
        self.near_slice=(slice(lo,hi),)*3
        self.region_masks=[self.valid & np.all(bands==np.array(r)[:,None,None,None],axis=0) for r in REGIONS]
        self.denoms=np.array([m.sum() for m in self.region_masks])

    def voxelize(self, directions, ranges, hit, scanned):
        c=self.c; d=np.asarray(directions,dtype=float).reshape(-1,3); r=np.asarray(ranges,dtype=float).ravel()
        h=np.asarray(hit,dtype=bool).ravel(); s=np.asarray(scanned,dtype=bool).ravel()
        if not (len(d)==len(r)==len(h)==len(s)): raise ValueError('shape mismatch')
        if np.any(h & ~s): raise ValueError('hit on unscanned ray')
        if np.any(s & ~h & (np.abs(r-self.c.max_range)>1e-5)): raise ValueError('scanned no-hit must be max range')
        if np.any(~np.isfinite(d)) or np.any(np.abs(np.linalg.norm(d,axis=1)-1)>1e-5): raise ValueError('unit directions required')
        if np.any(s & (~np.isfinite(r) | (r<c.start_range-1e-7) | (r>c.max_range+1e-5))): raise ValueError('invalid scanned range')
        grid=np.zeros((c.n,)*3,np.uint8); origin=np.array([0.,0.,c.origin_height]); eps=1e-10
        # DDA: positive-length traversal only, simultaneous axis ties advance together.
        for direction,dist,is_hit,valid in zip(d,r,h,s):
            if not valid: continue
            dist=max(c.start_range,min(float(dist),c.max_range))
            start=origin+direction*(c.start_range+eps)
            cell=np.floor((start+self.half)/c.voxel).astype(int)
            step=np.sign(direction).astype(int)
            boundary=-self.half+(cell+(step>0))*c.voxel
            tmax=np.full(3,np.inf); tdelta=np.full(3,np.inf); nz=step!=0
            tmax[nz]=(boundary[nz]-origin[nz])/direction[nz]
            tdelta[nz]=c.voxel/np.abs(direction[nz])
            endpoint=origin+direction*dist
            endcell=np.floor((endpoint+self.half)/c.voxel).astype(int)
            while np.all((cell>=0)&(cell<c.n)):
                if is_hit and np.array_equal(cell,endcell): break
                tnext=float(tmax.min())
                if grid[tuple(cell)]!=2: grid[tuple(cell)]=1
                if tnext>=dist-eps: break
                axes=np.abs(tmax-tnext)<eps
                cell[axes]+=step[axes]; tmax[axes]+=tdelta[axes]
            if is_hit and np.all((endcell>=0)&(endcell<c.n)):
                grid[tuple(endcell)]=2
        return grid

    def features(self, grid):
        a=np.asarray(grid)
        if a.shape!=(self.c.n,)*3 or not np.isin(a,[0,1,2]).all(): raise ValueError('invalid grid')
        n=a[self.near_slice].ravel(order='C')
        near=np.stack((n==2,n==1),axis=-1).ravel().astype(np.float32)
        far=np.array([(np.count_nonzero((a==2)&m)/den if den else 0, np.count_nonzero((a==1)&m)/den if den else 0) for m,den in zip(self.region_masks,self.denoms)],np.float32).ravel()
        return np.concatenate((near,far))

    def encode(self,directions,ranges,hit,scanned):
        grid=self.voxelize(directions,ranges,hit,scanned)
        return self.features(grid),grid

def rv1_directions():
    az=np.arange(128)*2*np.pi/128; el=np.linspace(-np.pi/6,np.pi/6,32)
    a,e=np.meshgrid(az,el)
    return np.stack((np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)),axis=-1)

