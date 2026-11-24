"""A4 and direction-enhanced sensor vectors for the original RV1 control task."""
import numpy as np

from experiments.rv1_range_cnn.lidar import LidarConfig
from fast_multiscale import FastEncoder

SCHEMA = {"original": "rv1_msgp_original_320_v2", "directional": "rv1_msgp_directional_448_v1"}
DIMS = {"original": 320, "directional": 448}


class ObservationEncoder:
    def __init__(self, mode, lidar_config=LidarConfig()):
        if lidar_config != LidarConfig():
            raise ValueError("expected the frozen 32x128 RV1 LiDAR")
        self.mode = mode
        self.dim = DIMS[mode]
        self.lidar_config = lidar_config
        self.voxels = FastEncoder()
        self.directions = lidar_config.directions()
        axes = np.arange(41)-20
        lateral = np.select([axes < -2, axes < 0, axes == 0, axes <= 2], [0,1,2,3], default=4)
        vertical = np.select([axes < -2, axes <= 2], [0,1], default=2)
        x,y,z = np.meshgrid(lateral,lateral,vertical,indexing="ij")
        self.keep = self.voxels.valid & ~((x>=1)&(x<=3)&(y>=1)&(y<=3)&(z==1))
        self.region = (x*15+y*3+z)[self.keep]
        denominator = np.bincount(self.region,minlength=75)
        self.nonempty = denominator > 0
        self.denominator = denominator[self.nonempty]
        assert self.denominator.size == 66
        origin = np.array([0,0,lidar_config.origin_height_m])
        times = np.full(self.directions.shape, np.inf)
        np.divide(np.where(self.directions>=0,1.,-1.)-origin,self.directions,out=times,
                  where=np.abs(self.directions)>1e-12)
        self.near_exit = times.min(axis=-1)[10:22].reshape(12,16,8)

    def _directional_far(self, grid, ranges, scanned):
        selected = grid[self.keep]
        counts = [np.bincount(self.region[selected==i],minlength=75)[self.nonempty] for i in (2,1)]
        regions = (np.stack(counts,axis=1)/self.denominator[:,None]).astype(np.float32).ravel()
        bands = ranges[10:22].reshape(12,16,8)
        available = scanned[10:22].reshape(12,16,8) & (bands>=self.near_exit)
        count = available.sum(axis=(0,2))
        total_depth = np.where(available,bands,0).sum(axis=(0,2))
        avg = np.divide(total_depth,8*count,out=np.zeros(16),where=count!=0)
        open_count = (available & (bands>=3.5)).sum(axis=(0,2))
        opened = np.divide(open_count,count,out=np.zeros(16),where=count!=0)
        sectors = np.stack((count/96,avg,opened),axis=-1).astype(np.float32).ravel()
        return regions,sectors

    def __call__(self, obs):
        state=np.asarray(obs["state"],np.float32)
        if state.shape!=(18,) or not np.isfinite(state).all():
            raise ValueError("State18 required")
        ranges=obs["range_m"]
        scanned=obs["scanned_mask"].astype(bool,copy=False)
        if self.mode=="original":
            base,_=self.voxels.encode(self.directions,ranges,obs["hit_mask"],scanned)
            result=np.concatenate((state,base))
        else:
            grid=self.voxels.voxelize(self.directions,ranges,obs["hit_mask"],scanned)
            near=grid[self.voxels.near_slice].ravel(order="C")
            bits=np.stack((near==2,near==1),axis=-1).astype(np.float32).ravel()
            regions,sectors=self._directional_far(grid,ranges,scanned)
            result=np.concatenate((state,bits,regions,sectors))
        if result.shape!=(self.dim,) or result.dtype!=np.float32 or not np.isfinite(result).all():
            raise ValueError("invalid policy vector")
        return result
