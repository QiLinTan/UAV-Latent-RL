import numpy as np
import pytest
import torch
from experiments.rv2_perception_design_v1.density import common96_mask,CapacityMatchedCompactEncoder,design
from experiments.rv2_perception_design_v1.multiscale import radial_images,InputMultiscalePrototype
from experiments.rv1_range_cnn.models import RangeEncoder
from experiments.rv1_range_cnn.codec import RangeCodec

torch.set_num_threads(2)

def test_density_count_and_exact_parameter_match():
    assert common96_mask().sum()==96
    base=RangeEncoder();n=sum(p.numel() for p in base.parameters())
    for shape in ((6,16),(3,32)):
        enc=CapacityMatchedCompactEncoder(shape);assert sum(p.numel() for p in enc.parameters())==n
        x=torch.rand(2,3,*shape,requires_grad=True);enc(x).sum().backward();assert torch.isfinite(x.grad).all()
    d=design();assert not d['legacy96_is_subset'];assert max(d['legacy_nearest_grid_angular_error_deg'])>.9

def test_static_unscanned_pixels_not_free():
    mask=common96_mask();r=np.full((32,128),np.nan,np.float32);r[mask]=8;hit=np.zeros_like(mask)
    codec=RangeCodec((32,128),8);q,m=codec.encode(r,hit,mask);image=codec.policy_image(q,m)
    assert not image[:,~mask].any();assert np.all(image[2,mask]==1);assert np.all(image[0,mask]==1)

def test_visibility_and_occlusion_semantics():
    r=torch.tensor([[[1.,3.,8.,float('nan')]]]);hit=torch.tensor([[[True,True,False,False]]]);scan=torch.tensor([[[True,True,True,False]]])
    x=radial_images(r,hit,scan)
    # Near hit: far band fully occluded, never marked free.
    assert x[0,2,3,0,0]==0 and x[0,2,4,0,0]==1 and x[0,2,5,0,0]==1
    # Mid hit: near band fully free; mid fraction strictly between 0 and 1.
    assert x[0,0,3,0,1]==1 and 0<x[0,1,3,0,1]<1
    # Scanned no-hit: every band free. Unscanned: every band unknown.
    assert torch.all(x[0,:,3,0,2]==1) and torch.all(x[0,:,5,0,3]==1) and not x[0,:,3,0,3].any()
    assert x[0,0,1,0,0]==1 and x[0,1,1,0,1]==1

def test_endpoint_assignment_and_bad_masks():
    r=torch.tensor([[[1.5,4.,8.]]]);hit=torch.ones_like(r,dtype=torch.bool);scan=hit.clone();x=radial_images(r,hit,scan)
    assert torch.equal(x[0,:,1,0].sum(0),torch.ones(3))
    with pytest.raises(ValueError):radial_images(r,hit,torch.zeros_like(scan))

def test_multiscale_shapes_finite_backward_and_no_rv1_mutation():
    frozen=RangeEncoder();original={k:v.clone() for k,v in frozen.state_dict().items()}
    m=InputMultiscalePrototype();r=torch.full((2,32,128),3.);h=torch.ones_like(r,dtype=torch.bool);s=h.clone()
    o=m(r,h,s,torch.zeros(2,18));assert o['latent64'].shape==(2,64) and o['concat192'].shape==(2,192)
    assert o['spatial_features'].shape==(2,3,32,4,16) and o['near_detail_full_resolution'].shape==(2,8,32,128)
    assert o['gate_weights'] is None
    (o['latent64'].square().mean()+o['near_detail_full_resolution'].square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(torch.equal(original[k],v) for k,v in frozen.state_dict().items())

def test_band_preprocessing_is_angular_seam_equivariant():
    r=torch.rand(1,32,128)*7.9+.08;h=torch.ones_like(r,dtype=torch.bool);s=h.clone()
    a=radial_images(r,h,s);b=radial_images(torch.roll(r,8,-1),torch.roll(h,8,-1),torch.roll(s,8,-1))
    assert torch.equal(b,torch.roll(a,8,-1))

def test_actual_fixed96_scan_and_replay_contract(tmp_path):
    from experiments.rv2_perception_design_v1.environment import StaticDensityEnv,density_contract
    from experiments.rv1_range_cnn.provenance import contract
    from experiments.rv1_range_cnn.lidar import LidarConfig
    from experiments.rv1_range_cnn.replay import Replay
    sparse=StaticDensityEnv(density=96);dense=StaticDensityEnv(density=4096)
    try:
        a,ai=sparse.reset(seed=43);b,bi=dense.reset(seed=43);mask=common96_mask()
        assert sparse.observation_space.contains(a);np.testing.assert_array_equal(a['state'],b['state'])
        np.testing.assert_array_equal(a['range_m'][mask],b['range_m'][mask]);assert np.isnan(a['range_m'][~mask]).all()
        assert ai['range_ray_count']==96 and bi['range_ray_count']==4096
        n,ra,ta,ua,ia=sparse.step([.2,0,.1]);m,rb,tb,ub,ib=dense.step([.2,0,.1])
        assert (ra,ta,ua)==(rb,tb,ub);np.testing.assert_array_equal(n['state'],m['state'])
        assert sparse.lidar.scan_count==dense.lidar.scan_count==2
        assert sparse.lidar.batch_calls==2 and dense.lidar.batch_calls==8
        assert ia['lower_steps_executed']==8 and ia['physics_substeps_executed']==16
        base=contract(LidarConfig(),'density_unit');c=density_contract(base,96);r=Replay(2,c)
        r.add(a,[.2,0,.1],n,ra,ta,ua,timestamps=(0,1/15));image=r.sample(2)[0]['image']
        assert torch.isfinite(image).all()
        assert not image.flatten(2)[:,:,~torch.from_numpy(mask.ravel())].any()
        assert torch.all(image[:,2].sum((1,2))==96)
        p=tmp_path/'r.pt';r.save(p)
        with pytest.raises(ValueError):Replay.load(p,density_contract(base,4096))
    finally:sparse.close();dense.close()

def test_static_density_course_reset_parity():
    from experiments.rv2_perception_design_v1.environment import make_density_course_env
    from experiments.rv1_range_cnn.environment import make_course_env
    from experiments.direct_velocity_full_v1.task_factory import load_config,ROOT
    from envs.map_atlas.forest_pool import ForestPool
    pool=ForestPool(ROOT/load_config()['forest_pool_path']);a=make_density_course_env(pool,96);b=make_course_env(pool)
    try:
        x,xi=a.reset(seed=43);y,yi=b.reset(seed=43);np.testing.assert_array_equal(x['state'],y['state'])
        for key in ['map_pool_index','start_curriculum_source','density_stage','start_curriculum_episode_level']:assert xi[key]==yi[key]
        assert a.lidar.scan_count==1 and x['scanned_mask'].sum()==96
    finally:a.close();b.close()
