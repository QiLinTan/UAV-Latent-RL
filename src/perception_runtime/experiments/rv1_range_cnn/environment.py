"""15Hz dense sensing over the unchanged 120/240Hz dv0 execution path."""
import numpy as np
from gymnasium import spaces
from experiments.direct_velocity_full_v1.envs.direct_velocity_full import NavigationPlant,DirectVelocityFullEnv
from experiments.direct_velocity_full_v1.observations.local_nav_state import local_state
from experiments.direct_velocity_full_v1.controls.velocity_tracking import VelocityTrackingController
from experiments.direct_velocity_full_v1.contracts import ControlSpec
from experiments.direct_velocity_full_v1.task_factory import load_config,gate_a_check,make_reward
from gym_pybullet_drones.utils.enums import DroneModel,Physics,ObservationType,ActionType
from .lidar import Lidar,LidarConfig

def plant_kwargs(map_spec=None):
    config=load_config()
    kwargs=dict(drone_model=DroneModel.CF2X,physics=Physics.PYB,pyb_freq=240,ctrl_freq=120,gui=False,
                record=False,obs=ObservationType.KIN,act=ActionType.RPM,range_azimuth_count=32,
                range_elevation_degrees=(-30.,0.,30.),curriculum=False,route_count_curriculum=False,
                curriculum_simple_mix_probability=0.,reward_model=make_reward(config),
                num_trees=0,route_blocking_tree=False)
    if map_spec is None:kwargs.update(initial_xyzs=np.array([[-3.5,0,1]]),target_pos=[3.5,0,1],episode_len_sec=20.)
    else:kwargs['map_spec']=map_spec
    return kwargs

class StateOnlyPlant(NavigationPlant):
    def _observationSpace(self):return spaces.Box(-4.,4.,shape=(1,18),dtype=np.float32)
    def _computeObs(self):
        state=self._getDroneStateVector(0)
        return local_state(state,self.TARGET_POS,self.accepted_world).astype(np.float32).reshape(1,18)

class RangeImageEnv(DirectVelocityFullEnv):
    def __init__(self,map_spec=None,lidar_config=None,**inherited_kwargs):
        config=load_config();gate_a_check(config)
        self.lidar=Lidar(lidar_config or LidarConfig())
        kwargs=inherited_kwargs.pop('plant_kwargs',None) or plant_kwargs(map_spec)
        control=inherited_kwargs.pop('control_spec',None) or ControlSpec(**config['control_spec'])
        if inherited_kwargs:raise TypeError('unknown constructor arguments')
        if control!=ControlSpec(**config['control_spec']):raise ValueError('frozen controller required')
        self.base_env=StateOnlyPlant(**kwargs)
        self.controller=VelocityTrackingController(spec=control)
        self.action_space=spaces.Box(-1.,1.,shape=(1,3),dtype=np.float32)
        c=self.lidar.config
        self.observation_space=spaces.Dict(dict(state=spaces.Box(-4.,4.,shape=(18,),dtype=np.float32),
            range_m=spaces.Box(0.,c.max_range_m,shape=c.shape,dtype=np.float32),
            hit_mask=spaces.MultiBinary(c.shape),scanned_mask=spaces.MultiBinary(c.shape)))
        self.active=False;self.upper_tick=0;self.last_ticks=[];self._defer_scan=False

    def _observation(self,state):
        if self._defer_scan:return state
        b=self.base_env
        scan=self.lidar.scan(client_id=b.CLIENT,position=b.pos[0],quaternion_xyzw=b.quat[0],
            timestamp_s=b.physics_ticks/240.,cache_key=self.upper_tick,self_ids=tuple(map(int,b.DRONE_IDS)),
            ground_ids=(int(b.PLANE_ID),) if hasattr(b,'PLANE_ID') else (),obstacle_ids=tuple(b._tree_ids))
        return dict(state=np.asarray(state).reshape(18).copy(),range_m=scan.range_m.copy(),
                    hit_mask=scan.hit_mask.copy(),scanned_mask=scan.scanned_mask.copy())

    def reset(self,*,seed=None,options=None):
        self.lidar.reset()
        state,info=super().reset(seed=seed,options=options)
        observation=self._observation(state)
        return observation,{**info,**self.sensor_info()}

    def sensor_info(self):
        c=self.lidar.config
        return dict(range_ray_count=c.height*c.width,legacy_unused_range_ray_count=int(self.base_env.NUM_RANGE_RAYS),
            lidar_scan_shape=list(c.shape),lidar_scans=self.lidar.scan_count,lidar_batches=self.lidar.batch_calls,
            lidar_timestamp_s=self.lidar.cache.timestamp_s if self.lidar.cache is not None else None)

    def step(self,action):
        state,r,t,u,info=super().step(action)
        observation=self._observation(state)
        return observation,r,t,u,{**info,**self.sensor_info()}

# The original course methods execute unchanged. MRO replaces only plant/sensing
# construction and defers reset sensing until the frozen course finishes relocation.
from experiments.direct_velocity_full_v1.envs.curriculum_adapter import CurriculumEnv
class RangeCurriculumEnv(CurriculumEnv,RangeImageEnv):
    def reset(self,*,seed=None,options=None):
        self._defer_scan=True
        try:state,info=CurriculumEnv.reset(self,seed=seed,options=options)
        finally:self._defer_scan=False
        observation=self._observation(state)
        return observation,{**info,**self.sensor_info()}

def make_course_env(pool,*,training=True,lidar_config=None):
    return RangeCurriculumEnv(pool=pool,training_course=training,split='train' if training else 'validation',
        lidar_config=lidar_config,control_spec=ControlSpec(**load_config()['control_spec']),
        plant_kwargs=plant_kwargs(pool.spec('train' if training else 'validation',0,0)))
