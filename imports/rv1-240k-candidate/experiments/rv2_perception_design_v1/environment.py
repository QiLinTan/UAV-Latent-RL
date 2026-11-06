"""Fixed96 vs4096 wrappers only; current rv1 formal run never imports this."""
import copy,hashlib
import numpy as np
from gymnasium import spaces
from experiments.rv1_range_cnn.environment import RangeImageEnv,RangeCurriculumEnv,plant_kwargs
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.direct_velocity_full_v1.task_factory import load_config
from experiments.direct_velocity_full_v1.contracts import ControlSpec
from .density import common96_mask

class RangeWithUnscannedSpace(spaces.Box):
    """Raw sensor NaNs denote unscanned only; codec yields finite CNN inputs."""
    def contains(self,x):
        x=np.asarray(x)
        return x.shape==self.shape and np.can_cast(x.dtype,self.dtype) and bool((np.isnan(x)|(np.isfinite(x)&(x>=0)&(x<=self.high))).all())

def density_contract(base,density):
    if density not in (96,4096):raise ValueError('static density')
    result=copy.deepcopy(base);mask=common96_mask() if density==96 else np.ones((32,128),bool)
    result['static_sampling']=dict(version='common_grid_density_v1',actual_rays=density,canvas_shape=[32,128],
        mask_sha256=hashlib.sha256(np.packbits(mask,bitorder='little').tobytes()).hexdigest(),
        missing_semantics='NaN raw distance, scanned=0, hit=0; finite all-zero policy pixels')
    return result

class StaticScanMixin:
    def __init__(self,*args,density=96,**kwargs):
        if density not in (96,4096):raise ValueError('density')
        self.static_mask=common96_mask() if density==96 else np.ones((32,128),bool)
        super().__init__(*args,**kwargs)
        if self.lidar.config!=LidarConfig():raise ValueError('common-grid density protocol fixes complete sensor geometry')
        self.observation_space.spaces['range_m']=RangeWithUnscannedSpace(0.,8.,shape=(32,128),dtype=np.float32)
    def _observation(self,state):
        if self._defer_scan:return state
        b=self.base_env
        scan=self.lidar.scan(client_id=b.CLIENT,position=b.pos[0],quaternion_xyzw=b.quat[0],timestamp_s=b.physics_ticks/240,
            cache_key=self.upper_tick,scanned_mask=self.static_mask,self_ids=tuple(map(int,b.DRONE_IDS)),
            ground_ids=(int(b.PLANE_ID),),obstacle_ids=tuple(b._tree_ids))
        return dict(state=np.asarray(state).reshape(18).copy(),range_m=scan.range_m.copy(),hit_mask=scan.hit_mask.copy(),scanned_mask=scan.scanned_mask.copy())
    def sensor_info(self):
        info=super().sensor_info();info.update(range_ray_count=int(self.static_mask.sum()),range_canvas_pixels=4096,
            static_scan_mask_sha256=hashlib.sha256(np.packbits(self.static_mask,bitorder='little').tobytes()).hexdigest())
        return info

class StaticDensityEnv(StaticScanMixin,RangeImageEnv):pass
class StaticDensityCourseEnv(StaticScanMixin,RangeCurriculumEnv):pass

def make_density_course_env(pool,density,training=True):
    split='train' if training else 'validation'
    return StaticDensityCourseEnv(pool=pool,training_course=training,split=split,density=density,
        lidar_config=LidarConfig(),control_spec=ControlSpec(**load_config()['control_spec']),plant_kwargs=plant_kwargs(pool.spec(split,0,0)))
