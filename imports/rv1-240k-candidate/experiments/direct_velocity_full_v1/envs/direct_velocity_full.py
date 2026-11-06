"""114D navigation through the accepted position-free Full velocity controller."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.forest.core import CustomForestAviary
from models.motor_action_codec import MotorActionCodec,MotorPhysicalLimits,ASYMMETRIC_RPM
from ..contracts import ControlState,VelocityCommand,normalized_action_to_world,yaw_from_quaternion
from ..controls.velocity_tracking import VelocityTrackingController
from ..observations.local_nav_state import local_state
from ..observations.range96_adapter import yaw_ray_geometry


class NavigationPlant(CustomForestAviary):
    def __init__(self,**kwargs):
        self.accepted_world=np.zeros(3);self.physics_ticks=0;self.ray_casts=0
        super().__init__(**kwargs)
        if self.NUM_RANGE_RAYS!=96 or self.CTRL_FREQ!=120 or self.PYB_FREQ!=240:
            raise ValueError('frozen 96 / 120 / 240 contract')
        self.set_motor_action_codec(MotorActionCodec(MotorPhysicalLimits(
            min_rpm=0,hover_rpm=self.HOVER_RPM,max_rpm=self.MAX_RPM,kf=self.KF),mode=ASYMMETRIC_RPM))

    def _observationSpace(self):
        return spaces.Box(-4.,4.,shape=(1,114),dtype=np.float32)

    def _rangeRayGeometry(self,pos):
        return yaw_ray_geometry(self,pos)

    def _computeObs(self):
        self.ray_casts+=1
        state=self._getDroneStateVector(0)
        return np.r_[local_state(state,self.TARGET_POS,self.accepted_world),
                     self._computeRangeObservation(state[:3])].astype(np.float32).reshape(1,114)

    def _physics(self,rpm,nth_drone):
        super()._physics(rpm,nth_drone);self.physics_ticks+=1

    def reset(self,**kwargs):
        self.accepted_world=np.zeros(3);self.physics_ticks=0;self.ray_casts=0
        return super().reset(**kwargs)


class DirectVelocityFullEnv(gym.Env):
    def __init__(self,*,control_spec,plant_kwargs):
        self.base_env=NavigationPlant(**plant_kwargs)
        self.controller=VelocityTrackingController(spec=control_spec)
        self.action_space=spaces.Box(-1.,1.,shape=(1,3),dtype=np.float32)
        self.observation_space=self.base_env.observation_space
        self.active=False;self.upper_tick=0;self.last_ticks=[]

    def motion_state(self):
        b=self.base_env
        return ControlState(b.quat[0],b.vel[0],b.ang_v[0])

    def initialize_control(self):
        b=self.base_env
        self.controller.reset(self.motion_state());self.upper_tick=0;self.active=True;self.last_ticks=[];self.episode_actions=[]
        b.accepted_world=np.zeros(3);b._last_motor_rpm=None;b._resetActionBuffer()

    def reset(self,*,seed=None,options=None):
        _,info=self.base_env.reset(seed=seed,options=options)
        self.initialize_control()
        return self.base_env._computeObs(),info

    def step(self,action):
        if not self.active:raise RuntimeError('reset required')
        a=np.asarray(action,dtype=np.float64)
        if a.shape not in ((3,),(1,3)) or not np.isfinite(a).all():raise ValueError('finite action (3,) or (1,3) required')
        raw=a.reshape(3).copy();b=self.base_env
        # Clip, project and rotate once at the decision boundary.
        command=normalized_action_to_world(raw,yaw_from_quaternion(b.quat[0]))
        from ..contracts import yaw_rotation
        projected=(yaw_rotation(yaw_from_quaternion(b.quat[0])).T@command)/[.8,.8,.3]
        packet=VelocityCommand(self.upper_tick,self.controller.tick,command)
        reward=0.;terms={};self.last_ticks=[];start_physics=b.physics_ticks
        for inner in range(8):
            rpm,d=self.controller.update(packet if inner==0 else None,self.motion_state())
            b.accepted_world=d['command_applied_world'].copy()
            motor=b.motor_action_codec.rpm_to_normalized_action(rpm).astype(np.float32)
            obs,r,terminated,truncated,info=b.step(motor.reshape(1,4))
            if not np.isfinite(obs).all() or not np.isfinite(r):raise FloatingPointError('observation/reward nonfinite')
            actual=b._last_motor_rpm.reshape(4)
            if not np.isfinite(actual).all() or np.any(actual<0) or np.any(actual>b.MAX_RPM+1e-6):raise FloatingPointError('illegal RPM')
            if np.max(abs(actual-rpm))>.005:raise RuntimeError('RPM codec mismatch')
            reward+=r
            for k,v in info.items():
                if k.startswith('reward/'):terms[k]=terms.get(k,0.)+v
            self.last_ticks.append(dict(lower_tick=self.controller.tick,physics_tick=b.physics_ticks,
                command_world=b.accepted_world.tolist(),velocity=b.vel[0].tolist(),position=b.pos[0].tolist(),
                rpy=b.rpy[0].tolist(),rpm=actual.tolist(),velocity_integral=self.controller.pi.integral.tolist(),
                attitude_integral=self.controller.adapter.integral.tolist(),
                force_saturated=bool(d['force_limited']),allocation_saturated=bool(d['allocation_saturated']),
                reward=float(r),done=bool(terminated or truncated),done_reason=info['done_reason']))
            if terminated or truncated:break
        self.upper_tick+=1
        self.episode_actions.append(raw.tolist())
        if b.physics_ticks-start_physics!=2*len(self.last_ticks):raise RuntimeError('physics/control timing mismatch')
        info={**info,**terms,'lower_steps_executed':len(self.last_ticks),'physics_substeps_executed':b.physics_ticks-start_physics,
              'raw_action':raw,'projected_action':projected.astype(np.float32),'accepted_world_command':b.accepted_world.copy()}
        if terminated or truncated:
            self.active=False;self.controller.finish_episode()
        return obs,float(reward),terminated,truncated,info

    def close(self):
        self.base_env.close()
