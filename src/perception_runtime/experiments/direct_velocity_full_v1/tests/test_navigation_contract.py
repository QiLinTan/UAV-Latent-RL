import copy
from unittest.mock import patch
import numpy as np
import pytest
import pybullet as p
from scipy.spatial.transform import Rotation
from envs.forest.core import CustomForestAviary
from envs.map_atlas.forest_pool import ForestPool
from ..task_factory import load_config,make_env,ROOT,make_reward
from ..contracts import yaw_rotation
from ..observations.local_nav_state import local_state


@pytest.fixture(scope='module')
def pool():return ForestPool(ROOT/load_config()['forest_pool_path'])


def test_state_schema_normalization_and_invariance():
    s=np.zeros(20);s[:3]=[1,2,1];s[3:7]=Rotation.from_euler('xyz',[.1,-.2,.4]).as_quat()
    s[7:10]=[.1,-.2,.4];s[10:13]=[.3,-.2,.1];s[13:16]=[.2,.4,-.3]
    goal=np.array([3,4,1.3]);command=np.array([.5,.1,-.1]);o=local_state(s,goal,command)
    assert o.shape==(18,) and o.dtype==np.float32 and np.isfinite(o).all()
    assert o[3]==pytest.approx(np.linalg.norm(goal-s[:3])/20)
    assert o[16]==pytest.approx((1-.08)/2.67)
    shift=np.array([5,-3,0]);t=s.copy();t[:3]+=shift
    np.testing.assert_allclose(o,local_state(t,goal+shift,command),atol=1e-6)
    r=yaw_rotation(1.2);t=s.copy();t[:3]=r@s[:3];t[3:7]=(Rotation.from_matrix(r)*Rotation.from_quat(s[3:7])).as_quat()
    t[10:13]=r@s[10:13];t[13:16]=r@s[13:16]
    np.testing.assert_allclose(o,local_state(t,r@goal,r@command),atol=1e-6)
    assert not local_state(s,s[:3],command)[:3].any()


def test_action_timing_new_controller_and_no_old_calls():
    env=make_env(empty=True)
    try:
        env.reset(seed=43);b=env.base_env
        p.resetBasePositionAndOrientation(int(b.DRONE_IDS[0]),b.pos[0],p.getQuaternionFromEuler([0,0,np.pi/2]),physicsClientId=b.CLIENT)
        b._updateAndStoreKinematicInformation();env.initialize_control()
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        with patch.object(DSLPIDControl,'computeControl',side_effect=AssertionError('old position PID used')):
            obs,r,t,u,info=env.step([[1,1,1]])
        assert obs.shape==(1,114) and info['lower_steps_executed']==8 and info['physics_substeps_executed']==16
        np.testing.assert_allclose(info['accepted_world_command'],[-.8/np.sqrt(2),.8/np.sqrt(2),.3],atol=1e-6)
        assert len(env.last_ticks)==8 and b.step_counter==16
        assert all(np.allclose(x['command_world'],info['accepted_world_command']) for x in env.last_ticks)
        assert r==pytest.approx(sum(x['reward'] for x in env.last_ticks))
        assert env.controller.tick==8
        assert not hasattr(env,'reference_position') and not hasattr(env,'last_reference_packet')
        snap=env.controller.snapshot();counter=b.step_counter
        with pytest.raises(ValueError):env.step([np.nan,0,0])
        assert snap==env.controller.snapshot() and counter==b.step_counter
    finally:env.close()


def test_rays_real_cast_yaw_zero_and_rotated_hits():
    env=make_env(empty=True)
    try:
        env.reset(seed=1);b=env.base_env;pos=b.pos[0].copy()
        got=b._rangeRayGeometry(pos);old=CustomForestAviary._rangeRayGeometry(b,pos)
        for x,y in zip(got,old):np.testing.assert_allclose(x,y,atol=3e-7)
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.1,.1,.4],physicsClientId=b.CLIENT)
        p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=pos+[0,.8,0],physicsClientId=b.CLIENT)
        # Local zero-azimuth ray, middle elevation ring.
        assert b._computeRangeObservation(pos)[32]>.95
        p.resetBasePositionAndOrientation(int(b.DRONE_IDS[0]),pos,p.getQuaternionFromEuler([.1,-.1,np.pi/2]),physicsClientId=b.CLIENT)
        b._updateAndStoreKinematicInformation()
        d=b._computeRangeObservation(pos)
        assert .3<d[32]<.5
        geom=b._rangeRayGeometry(pos)
        np.testing.assert_allclose((geom[1]-geom[0])[32,:2],[0,b.OBSTACLE_SENSOR_RANGE-b.RANGE_RAY_ORIGIN_OFFSET],atol=1e-6)
    finally:env.close()


def test_reward_done_exact_task_contract():
    env=make_env(empty=True);c=load_config()
    try:
        env.reset(seed=3);b=env.base_env
        assert type(b)._computeReward is CustomForestAviary._computeReward
        assert type(b)._classifyDoneReason is CustomForestAviary._classifyDoneReason
        for key,value in c['reward_kwargs'].items():
            if key=='integration_dt':assert value==1/120
            else:assert value==c['old_effective_config'].get(key,160. if key=='near_goal_velocity_tracking_weight' else None)
        rng=np.random.default_rng(3)
        for reason in ('running','success','collision','xy_bound','height_bound','attitude_bound','timeout'):
            s=b._getDroneStateVector(0).copy();s[10:13]=rng.normal(0,.2,3)
            x=dict(state=s,prev_goal_dist=7.1,prev_pos=s[:3]+[.01,0,0],start_pos=b.START_POS,target_pos=b.TARGET_POS,
                goal_tolerance=b.GOAL_TOLERANCE,safe_distance=b.SAFE_DISTANCE,clearance=.2,collision=reason=='collision',terminal_reason=reason)
            a=b._reward_model.compute(**x);z=make_reward(c).compute(**x)
            assert a==z
            if reason!='running':
                expected={'success':250.,'collision':-300.,'xy_bound':-250.,'height_bound':-250.,'attitude_bound':-250.,'timeout':-180.}
                assert a[0]==expected[reason]
        s=b._getDroneStateVector(0).copy();s[:3]=b.TARGET_POS
        assert b._classifyDoneReason(s,collision=True)=='success'
        # Preserve original float32 conversion: representable .08 is slightly below the Python threshold.
        s[:3]=[0,0,.08];assert b._classifyDoneReason(s,collision=False)==CustomForestAviary._classifyDoneReason(b,s,collision=False)
        s[2]=np.nextafter(np.float32(.08),np.float32(np.inf));assert b._classifyDoneReason(s,collision=False)=='running'
        s[2]=.079;assert b._classifyDoneReason(s,collision=False)=='height_bound'
        s[2]=1.;s[7]=.65;assert b._classifyDoneReason(s,collision=False)=='running'
        s[7]=.651;assert b._classifyDoneReason(s,collision=False)=='attitude_bound'
        b.EPISODE_LEN_SEC=0.
        # Original BaseAviary increments step_counter after reward/done: strict timeout at zero is observed on tick two.
        _,r,t,u,i=env.step([0,0,0]);assert u and not t and i['lower_steps_executed']==2
        assert b.step_counter==4 and not env.active
        with pytest.raises(RuntimeError):env.step([0,0,0])
    finally:env.close()


def test_curriculum_matches_old_pool_rng_candidates_and_reset(pool):
    from envs.forest.ego_curriculum_env import EgoCurriculumFullForestEnv
    from gym_pybullet_drones.utils.enums import Physics
    new=make_env(pool=pool)
    old=EgoCurriculumFullForestEnv(forest_pool_path=ROOT/load_config()['forest_pool_path'],
        pyb_freq=240,ctrl_freq=120,physics=Physics.PYB,gui=False,semantic_history_length=8,
        range_azimuth_count=32,range_elevation_degrees=(-30,0,30))
    try:
        for level in range(5):
            new.start_curriculum.level=level;old.start_curriculum.level=level
            for seed in (43,44):
                no,ni=new.reset(seed=seed);oo,oi=old.reset(seed=seed)
                assert ni['geometry_sha256']==oi['geometry_sha256']
                assert new._pool_index==old._pool_index and new._start_source==old._start_source
                np.testing.assert_allclose(new.base_env.pos,old.base_env.pos,atol=1e-6)
                assert new._pool_rng.bit_generator.state==old._pool_rng.bit_generator.state
                assert new._start_rng.bit_generator.state==old._start_rng.bit_generator.state
                if level<4:np.testing.assert_array_equal(new._start_candidates(new.start_curriculum.bands[level]),old._start_candidates(old.start_curriculum.bands[level]))
        new.step([.3,.2,0]);new.reset(seed=43)
        assert new.controller.tick==0 and new.upper_tick==0 and not new.controller.pi.integral.any()
        assert not new.controller.adapter.integral.any() and not new.base_env.accepted_world.any()
        assert new.base_env._last_motor_rpm is None
        assert new.base_env._prev_goal_dist==pytest.approx(np.linalg.norm(new.base_env.pos[0]-new.base_env.TARGET_POS))
    finally:new.close();old.close()


def test_script_and_random_smoke(pool):
    env=make_env(pool=pool);rng=np.random.default_rng(43);count=0
    try:
        obs,_=env.reset(seed=43)
        for step in range(140):
            action=np.clip(obs[0,:3]*.5,-1,1) if step<40 else rng.uniform(-1,1,3)
            obs,r,t,u,info=env.step(action);count+=info['lower_steps_executed']
            assert np.isfinite(obs).all() and np.isfinite(r)
            if t or u:obs,_=env.reset()
        assert count>800
    finally:env.close()
