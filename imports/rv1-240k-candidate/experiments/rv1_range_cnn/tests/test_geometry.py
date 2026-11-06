import numpy as np
import pybullet as p
import pytest
from experiments.rv1_range_cnn.lidar import LidarConfig,Lidar,LAYOUTS
from experiments.rv1_range_cnn.codec import RangeCodec
from experiments.rv1_range_cnn.environment import RangeImageEnv
from experiments.direct_velocity_full_v1.task_factory import make_env

@pytest.fixture
def client():
    c=p.connect(p.DIRECT)
    yield c
    p.disconnect(c)

def scan(sensor,c,**kwargs):
    return sensor.scan(client_id=c,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0,**kwargs)

def test_layout_and_resolution():
    assert [c.height*c.width for c in LAYOUTS.values()]==[96,512,1024,2048,4096,4096]
    assert LidarConfig.from_resolution(horizontal_resolution_deg=2.8125,vertical_resolution_deg=60/31)==LidarConfig()
    c=LidarConfig(height=3,width=5,horizontal_fov_deg=90)
    np.testing.assert_allclose(c.directions()[1,-1],[0,1,0],atol=1e-15)
    for cfg in LAYOUTS.values():np.testing.assert_allclose(np.linalg.norm(cfg.directions(),axis=-1),1,atol=1e-14)
    with pytest.raises(ValueError):LidarConfig(max_range_m=.01)
    with pytest.raises(ValueError):LidarConfig.from_resolution(horizontal_resolution_deg=7,vertical_resolution_deg=2)

def test_first_hit_unknown_and_partial_mask(client):
    near=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.1,.3,.3],physicsClientId=client)
    n=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=near,basePosition=[1,0,1.02],physicsClientId=client)
    p.createMultiBody(baseMass=0,baseCollisionShapeIndex=near,basePosition=[2,0,1.02],physicsClientId=client)
    cfg=LidarConfig(height=1,width=4,vertical_min_deg=0,vertical_max_deg=0)
    sensor=Lidar(cfg);mask=np.array([[True,True,False,True]])
    s=scan(sensor,client,scanned_mask=mask,obstacle_ids=[n],cache_key=0)
    assert s.hit_body[0,0]==n and s.hit_type[0,0]==3
    assert abs(s.range_m[0,0]-.9)<1e-6
    assert s.hit_mask[0,1]==False and s.range_m[0,1]==8
    assert not s.scanned_mask[0,2] and np.isnan(s.range_m[0,2])
    np.testing.assert_array_equal(s.visibility(0,0,[.01,.5,.9,1.5,3,9]),[0,1,2,0,0,0])
    np.testing.assert_array_equal(s.visibility(0,1,[.01,.5,7.9,9]),[0,1,1,0])
    assert not s.visibility(0,2,[1,2]).any()
    np.testing.assert_allclose(s.pointcloud(),s.raw_hit_world[s.hit_mask],atol=1e-6)
    assert scan(sensor,client,scanned_mask=mask,cache_key=0) is s
    assert sensor.scan_count==1
    with pytest.raises(ValueError):s.range_m[0,0]=3

def test_yaw_horizontal_not_body_tilt(client):
    c=LidarConfig(height=3,width=32);sensor=Lidar(c)
    q=p.getQuaternionFromEuler([.3,-.4,np.pi/2])
    s=sensor.scan(client_id=client,position=[1,2,1],quaternion_xyzw=q,timestamp_s=.2)
    np.testing.assert_allclose(s.directions_world[1,0],[0,1,0],atol=1e-15)
    np.testing.assert_allclose(s.origin_world,[1,2,1.02])
    assert s.timestamp_s==.2
    with pytest.raises(ValueError):sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,0],timestamp_s=0)
    assert sensor.scan_count==1

def test_ground_self_and_chunking(client):
    plane=p.createCollisionShape(p.GEOM_PLANE,physicsClientId=client)
    ground=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=plane,physicsClientId=client)
    c=LidarConfig(height=3,width=32,batch_size=17);a=scan(Lidar(c),client,ground_ids=[ground])
    assert np.sum(a.hit_type==2)>0
    np.testing.assert_allclose(a.pointcloud()[:,2],0,atol=2e-6)
    shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.2,physicsClientId=client)
    body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[.5,0,1.02],physicsClientId=client)
    sensor=Lidar(c);s=scan(sensor,client,self_ids=[body],ground_ids=[ground])
    assert (s.hit_type==1).any() and sensor.batch_calls==6
    other=scan(Lidar(LidarConfig(height=3,width=32,batch_size=1024)),client,self_ids=[body],ground_ids=[ground])
    np.testing.assert_array_equal(s.hit_mask,other.hit_mask);np.testing.assert_allclose(s.range_m,other.range_m)

@pytest.mark.parametrize('name',list(LAYOUTS))
def test_quantization(name):
    c=LAYOUTS[name];codec=RangeCodec(c.shape,c.max_range_m);rng=np.random.default_rng(43)
    r=rng.uniform(c.origin_offset_m,c.max_range_m,c.shape).astype(np.float32)
    hit=rng.random(c.shape)>.2;scanned=rng.random(c.shape)>.1;hit&=scanned;r[scanned & ~hit]=c.max_range_m;r[~scanned]=np.nan
    q,m=codec.encode(r,hit,scanned);r2,h2,s2=codec.decode(q,m)
    assert np.nanmax(abs(r-r2))<=codec.quantum_m/2+1e-6
    np.testing.assert_array_equal(hit,h2);np.testing.assert_array_equal(scanned,s2)
    image=codec.policy_image(q,m);assert image.shape==(3,*c.shape) and np.isfinite(image).all()

def test_15_120_240_cache_reset_and_task_parity():
    dense=RangeImageEnv();old=make_env(empty=True)
    try:
        obs,reset_info=dense.reset(seed=43);legacy,_=old.reset(seed=43)
        assert reset_info['range_ray_count']==4096 and reset_info['legacy_unused_range_ray_count']==96
        np.testing.assert_allclose(obs['state'],legacy.reshape(-1)[:18],atol=1e-7)
        for i in range(12):
            action=np.array([.3,-.1,.1])
            n,r,t,u,info=dense.step(action);o,ro,to,uo,io=old.step(action)
            assert (r,t,u,info['done_reason'])==(ro,to,uo,io['done_reason'])
            np.testing.assert_allclose(n['state'],o.reshape(-1)[:18],atol=1e-7)
            assert info['lower_steps_executed']==8 and info['physics_substeps_executed']==16
            assert info['range_ray_count']==4096 and info['lidar_scan_shape']==[32,128]
            assert dense.lidar.scan_count==i+2 and dense.base_env.ray_casts==0
            assert dense.lidar.cache.timestamp_s==(i+1)*16/240
            assert not (dense.lidar.cache.hit_type==1).any()
        dense.reset(seed=43)
        assert dense.controller.tick==0 and dense.lidar.scan_count==1
        assert np.linalg.norm(dense.controller.pi.integral)==0
        assert not dense.base_env.accepted_world.any()
        assert dense.lidar.cache.timestamp_s==0
    finally:dense.close();old.close()
