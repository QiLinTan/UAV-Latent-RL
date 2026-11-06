import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv2_multiscale_range.contracts import PerceptionSpec, make_contract, validate_contract, SENSOR
from experiments.rv2_multiscale_range.features import canonical_image, radial_images, NearBypass, sector_indices

torch.set_num_threads(1)

def raw():
    return dict(state=np.zeros(18, np.float32), range_m=np.full((32,128), 8., np.float32),
                hit_mask=np.zeros((32,128), bool), scanned_mask=np.ones((32,128), bool))

def image(obs):
    return torch.from_numpy(canonical_image(obs)).unsqueeze(0)

def test_blindness_nohit_and_occlusion():
    o=raw(); o['hit_mask'][16,0]=True; o['range_m'][16,0]=.8
    o['scanned_mask'][16,1]=False; o['range_m'][16,1]=np.nan
    b=radial_images(image(o))
    assert b.shape==(1,3,5,32,128)
    assert b[0,0,0,16,0]==1 and b[0,1,3,16,0]==1 and b[0,2,3,16,0]==1
    assert b[0,1,2,16,0]==0 and b[0,2,2,16,0]==0
    assert abs(float(b[0,0,2,0,0])-(1.5-.08)/1.5)<1e-6
    assert b[0,1,2,0,0]==1 and b[0,2,2,0,0]==1
    assert not b[0,:,:,16,1].any()  # scanned channel distinguishes unknown from measured free.
    assert torch.isfinite(b).all()

@pytest.mark.parametrize('value', [.08, .8, 1.5-1e-4, 1.5, 1.5+1e-4, 4.-1e-4, 4., 4.+1e-4, 8.])
def test_quantized_threshold_unique_membership(value):
    o=raw();o['hit_mask'][15,0]=True;o['range_m'][15,0]=value
    x=image(o);d=float(x[0,0,15,0]*8)
    b=radial_images(x)
    assert b[0,:,0,15,0].sum()==1
    i=0 if d<1.5 else 1 if d<4 else 2
    assert b[0,i,0,15,0]==1
    bypass=NearBypass(LidarConfig().directions())(x).reshape(24,2)
    assert bool((bypass[:,0]<1).any())==(d<1.5)

def test_exact_mathematical_endpoints():
    o=image(raw());o[0,0,0,:3]=torch.tensor([1.5,4.,8.])/8;o[0,1,0,:3]=1
    b=radial_images(o)
    assert b[0,:,0,0,:3].tolist()==[[0.,0.,0.],[1.,0.,0.],[0.,1.,1.]]

def test_bypass_nearest_valid_sector_and_limits():
    o=raw();spec=PerceptionSpec(bypass=True);idx=sector_indices(LidarConfig().directions(),spec)
    module=NearBypass(LidarConfig().directions(),spec)
    empty=module(image(o)).reshape(24,2);assert torch.equal(empty,torch.ones_like(empty))
    o['hit_mask'][15,0]=True;o['range_m'][15,0]=.8
    a=module(image(o)).reshape(24,2);sector=idx[15,0]
    assert (a[:,0]!=empty[:,0]).sum()==1 and a[sector,0]<1
    o['hit_mask'][15,1]=True;o['range_m'][15,1]=.4
    b=module(image(o));assert b.reshape(24,2)[sector,0]<a[sector,0]
    # Counterexample: another farther return in same sector is intentionally lost.
    o['hit_mask'][15,2]=True;o['range_m'][15,2]=.9
    assert torch.equal(b,module(image(o)))
    o['scanned_mask'][:]=False;o['hit_mask'][:]=False;o['range_m'][:]=np.nan
    unknown=module(image(o)).reshape(24,2)
    assert torch.equal(unknown[:,0],torch.ones(24)) and not unknown[:,1].any()

def test_partial_sector_coverage():
    spec=PerceptionSpec(bypass=True);directions=LidarConfig().directions();idx=sector_indices(directions,spec)
    o=raw();o['scanned_mask'][:]=False;o['range_m'][:]=np.nan
    o['scanned_mask'][15,0]=True;o['range_m'][15,0]=8
    v=NearBypass(directions,spec)(image(o)).reshape(24,2)
    assert abs(float(v[idx[15,0],1])-1/np.sum(idx==idx[15,0]))<1e-7
    assert torch.all(v[:,0]==1)

def test_angles_seam_and_elevation_edges():
    cfg=LidarConfig();directions=cfg.directions();s=PerceptionSpec(bypass=True)
    ids=sector_indices(directions,s)
    assert ids[15,0]==ids[15,-1]  # front sector spans 360/0.
    assert np.bincount(ids.ravel(),minlength=24).min()>0
    # Construct directions exactly on +/-22.5deg and vertical +/-10deg edges.
    for az,el,h,v in [(-22.5,-10,0,1),(22.5,10,1,2),(0,-30,0,0),(0,30,0,2)]:
        d=np.array(directions,copy=True);a,e=np.deg2rad([az,el]);d[0,0]=[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]
        assert sector_indices(d,s)[0,0]==v*8+h

def test_radial_roll_equivariance_and_bypass_sector_roll():
    o=raw();o['range_m'][12,0]=.4;o['hit_mask'][12,0]=True;x=image(o)
    assert torch.equal(radial_images(x.roll(16,-1)),radial_images(x).roll(16,-1))
    module=NearBypass(LidarConfig().directions())
    a=module(x).reshape(1,3,8,2);b=module(x.roll(16,-1)).reshape(1,3,8,2)
    assert torch.equal(a.roll(1,2),b)

@pytest.mark.parametrize('sectors,edges,dim',[(4,(-30,-10,10,30),24),(8,(-30,0,30),32),(8,(-30,-10,10,30),48),(16,(-30,-10,10,30),96)])
def test_bypass_capacity_interfaces(sectors,edges,dim):
    spec=PerceptionSpec(bypass=True,horizontal_sectors=sectors,vertical_edges_deg=edges)
    assert spec.bypass_dim==dim
    assert NearBypass(LidarConfig().directions(),spec)(image(raw())).shape==(1,dim)

def test_invalid_inputs_do_not_mutate_buffers():
    module=NearBypass(LidarConfig().directions());before=copy.deepcopy(module.state_dict())
    for channel,value in [(0,float('nan')),(1,.5),(2,-1)]:
        x=image(raw());x[0,channel,0,0]=value
        with pytest.raises(ValueError):module(x)
    assert all(torch.equal(before[k],v) for k,v in module.state_dict().items())
    o=raw();o['range_m'][0,0]=.01;o['hit_mask'][0,0]=True
    with pytest.raises(ValueError):canonical_image(o)

def test_contract_rejection():
    parent=dict(sensor=SENSOR,state_dim=18)
    c=make_contract(PerceptionSpec(bypass=True),parent,'test')
    assert c['policy_input_dim']==130
    for bad in [dict(schema='rv1'),make_contract(PerceptionSpec(),parent,'test'),make_contract(PerceptionSpec(bypass=True),parent,'changed')]:
        with pytest.raises(ValueError):validate_contract(bad,c)
    with pytest.raises(ValueError):PerceptionSpec(mode='near_fine_far_coarse')
    with pytest.raises(ValueError):make_contract(PerceptionSpec(),dict(sensor={**SENSOR,'max_range_m':1.75},state_dim=18),'test')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CPU-only host: GPU consistency unverified')
def test_cpu_gpu_bypass_consistency():
    o=raw();o['hit_mask'][12,0]=True;o['range_m'][12,0]=.4;x=image(o)
    module=NearBypass(LidarConfig().directions())
    torch.testing.assert_close(module(x),module.cuda()(x.cuda()).cpu(),rtol=0,atol=1e-7)
