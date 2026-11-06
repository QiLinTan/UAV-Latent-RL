"""First-return yaw-horizontal LiDAR. No map/route truth enters policy data."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pybullet as p
from experiments.direct_velocity_full_v1.contracts import yaw_rotation,yaw_from_quaternion

@dataclass(frozen=True)
class LidarConfig:
    height:int=32
    width:int=128
    horizontal_fov_deg:float=360.
    vertical_min_deg:float=-30.
    vertical_max_deg:float=30.
    azimuth_start_deg:float=0.
    max_range_m:float=8.
    origin_offset_m:float=.08
    origin_height_m:float=.02
    batch_size:int=1024

    def __post_init__(self):
        values=[v for v in asdict(self).values()]
        if not all(np.isfinite(v) for v in values):raise ValueError('nonfinite geometry')
        if any(type(v) is not int or v<1 for v in (self.height,self.width,self.batch_size)):raise ValueError('positive integer dimensions')
        if not 0<self.horizontal_fov_deg<=360 or not -90<=self.vertical_min_deg<=self.vertical_max_deg<=90:raise ValueError('FOV')
        if self.height>1 and self.vertical_min_deg==self.vertical_max_deg:raise ValueError('duplicate elevations')
        if not 0<=self.origin_offset_m<self.max_range_m:raise ValueError('range/offset')
        if self.batch_size>p.MAX_RAY_INTERSECTION_BATCH_SIZE:raise ValueError('PyBullet batch limit')

    @classmethod
    def from_resolution(cls,*,horizontal_resolution_deg,vertical_resolution_deg,**kwargs):
        fov=kwargs.get('horizontal_fov_deg',360.);span=kwargs.get('vertical_max_deg',30.)-kwargs.get('vertical_min_deg',-30.)
        if min(horizontal_resolution_deg,vertical_resolution_deg)<=0:raise ValueError('positive resolution')
        intervals=[fov/horizontal_resolution_deg,span/vertical_resolution_deg]
        if any(abs(v-round(v))>1e-8 for v in intervals):raise ValueError('FOV must divide exactly by resolution')
        return cls(width=round(intervals[0])+(fov!=360.),height=round(intervals[1])+1,**kwargs)

    @property
    def circular(self):return self.horizontal_fov_deg==360.
    @property
    def shape(self):return self.height,self.width
    @property
    def angular_resolution_deg(self):
        return (self.horizontal_fov_deg/(self.width if self.circular else max(1,self.width-1)),
                (self.vertical_max_deg-self.vertical_min_deg)/max(1,self.height-1))
    def directions(self):
        az=np.deg2rad(self.azimuth_start_deg+np.linspace(0,self.horizontal_fov_deg,self.width,endpoint=not self.circular))
        el=np.deg2rad(np.linspace(self.vertical_min_deg,self.vertical_max_deg,self.height))
        a,e=np.meshgrid(az,el)
        return np.stack((np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)),axis=-1)

LAYOUTS={name:LidarConfig(height=h,width=w) for name,h,w in
         [('96_3x32',3,32),('512_8x64',8,64),('1024_16x64',16,64),('2048_16x128',16,128),
          ('4096_32x128',32,128),('4096_16x256',16,256)]}

@dataclass(frozen=True)
class Scan:
    config:LidarConfig
    timestamp_s:float
    origin_world:np.ndarray
    directions_world:np.ndarray
    range_m:np.ndarray
    hit_mask:np.ndarray
    scanned_mask:np.ndarray
    hit_body:np.ndarray  # QA only; never included in policy observation
    hit_type:np.ndarray  # QA only: 0 none, 1 self, 2 ground, 3 obstacle, 4 other
    raw_hit_world:np.ndarray

    def pointcloud(self):
        valid=self.hit_mask & self.scanned_mask
        return self.origin_world+self.directions_world[valid]*self.range_m[valid,None]

    def visibility(self,row,col,radial_distance_m,tolerance=1e-5):
        """0 unknown/blind/occluded, 1 ray-visible free, 2 first surface.

        This is a query ON a ray, never a claim about the angular gap between rays.
        """
        d=np.asarray(radial_distance_m);r=self.range_m[row,col]
        known=self.scanned_mask[row,col] & (d>=self.config.origin_offset_m) & (d<=self.config.max_range_m)
        out=np.zeros(d.shape,np.uint8)
        out[known & (d<r-tolerance)]=1
        if self.hit_mask[row,col]:out[known & (abs(d-r)<=tolerance)]=2
        else:out[known & (d<=r)]=1
        return out

class Lidar:
    def __init__(self,config):
        self.config=config;self.directions_local=config.directions();self.reset()

    def reset(self):
        self.cache=None;self.cache_key=None;self.scan_count=0;self.batch_calls=0

    def scan(self,*,client_id,position,quaternion_xyzw,timestamp_s,cache_key=None,
             scanned_mask=None,self_ids=(),ground_ids=(),obstacle_ids=()):
        c=self.config;pos=np.asarray(position,dtype=float);q=np.asarray(quaternion_xyzw,dtype=float)
        if pos.shape!=(3,) or q.shape!=(4,) or not np.isfinite(pos).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5 or not np.isfinite(timestamp_s) or timestamp_s<0:
            raise ValueError('finite SI pose, unit xyzw quaternion and nonnegative simulation timestamp required')
        mask=np.ones(c.shape,bool) if scanned_mask is None else np.asarray(scanned_mask,dtype=bool)
        if mask.shape!=c.shape:raise ValueError('scan mask shape')
        if cache_key is not None and cache_key==self.cache_key:
            if not np.array_equal(mask,self.cache.scanned_mask):raise ValueError('cache key reused for different scan mask')
            return self.cache
        directions=self.directions_local @ yaw_rotation(yaw_from_quaternion(q)).T
        origin=pos+[0,0,c.origin_height_m]
        starts=origin+directions*c.origin_offset_m;ends=origin+directions*c.max_range_m
        ids=np.flatnonzero(mask.ravel());results=[];calls=0
        for at in range(0,len(ids),c.batch_size):
            sub=ids[at:at+c.batch_size]
            results.extend(p.rayTestBatch(starts.reshape(-1,3)[sub].tolist(),ends.reshape(-1,3)[sub].tolist(),
                                         numThreads=1,physicsClientId=client_id));calls+=1
        distances=np.full(c.height*c.width,np.nan,np.float32)
        body=np.full(len(distances),-1,np.int32);types=np.zeros(len(distances),np.uint8)
        xyz=np.full((len(distances),3),np.nan,np.float64)
        for i,res in zip(ids,results):
            body[i]=res[0];distances[i]=c.origin_offset_m+res[2]*(c.max_range_m-c.origin_offset_m)
            if res[0]>=0:
                xyz[i]=res[3];types[i]=1 if res[0] in self_ids else 2 if res[0] in ground_ids else 3 if res[0] in obstacle_ids else 4
        scan=Scan(c,float(timestamp_s),origin.copy(),directions,distances.reshape(c.shape),
                  (body>=0).reshape(c.shape),mask.copy(),body.reshape(c.shape),types.reshape(c.shape),xyz.reshape(*c.shape,3))
        for a in (scan.origin_world,scan.directions_world,scan.range_m,scan.hit_mask,scan.scanned_mask,scan.hit_body,scan.hit_type,scan.raw_hit_world):a.setflags(write=False)
        self.cache=scan;self.cache_key=cache_key;self.scan_count+=1;self.batch_calls+=calls
        return scan
