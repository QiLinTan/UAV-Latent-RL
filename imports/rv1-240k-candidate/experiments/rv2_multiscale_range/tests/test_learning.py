import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from experiments.rv2_multiscale_range.contracts import PerceptionSpec,make_contract,SENSOR
from experiments.rv2_multiscale_range.features import canonical_image,radial_images,NearBypass
from experiments.rv2_multiscale_range.models import Actor,Critic,Encoder
from experiments.rv2_multiscale_range.replay import Replay,SuccessReplay
from experiments.rv2_multiscale_range.td3 import RangeTD3
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.direct_velocity_full_v1.task_factory import load_config

torch.set_num_threads(1)

def raw():
    return dict(state=np.zeros(18,np.float32),range_m=np.full((32,128),8,np.float32),hit_mask=np.zeros((32,128),bool),scanned_mask=np.ones((32,128),bool))

def contract(spec):
    return make_contract(spec,dict(sensor=SENSOR,state_dim=18),'tests')

@pytest.mark.parametrize('kind,bypass',[('single',False),('single',True),('multiscale',False),('multiscale',True),('joint_bands',False),('single_large',False)])
def test_forward_backward_shapes_and_encoder_independence(kind,bypass):
    s=PerceptionSpec(encoder=kind,bypass=bypass)
    x=dict(state=torch.zeros(2,18),image=torch.from_numpy(canonical_image(raw())).unsqueeze(0).repeat(2,1,1,1))
    a,c=Actor(s),Critic(s);action=a(x);q1,q2=c(x,action)
    assert action.shape==(2,3) and q1.shape==q2.shape==(2,1)
    assert a.features.input_dim==82+s.bypass_dim
    loss=action.square().sum()+q1.square().sum()+q2.square().sum();loss.backward()
    for m in (a,c):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert not set(map(id,a.features.encoder.parameters())) & set(map(id,c.features.encoder.parameters()))
    z,detail=a.features.encoder(x['image'],True);assert z.shape==(2,64)
    if kind=='multiscale':
        assert detail['early'][0].shape==(2,12,32,128)
        assert detail['spatial'][0].shape==(2,30,8,32)

def test_capacity():
    count=lambda m:sum(p.numel() for p in m.parameters())
    single=count(Encoder(PerceptionSpec(encoder='single')))
    multi=count(Encoder(PerceptionSpec()))
    assert single==145472 and abs(multi/single-1)<.03
    a=Actor(PerceptionSpec());b=Actor(PerceptionSpec(bypass=True))
    assert count(b)-count(a)==48*256

def test_online_replay_identical_at_quantization_edges(tmp_path):
    spec=PerceptionSpec(bypass=True);c=contract(spec);r=Replay(16,c)
    for i,d in enumerate([.08,.3,1.5-1e-5,1.5,1.5+1e-5,4-1e-5,4.,4+1e-5,8.]):
        o=raw();o['range_m'][12,0]=d;o['hit_mask'][12,0]=True
        o['scanned_mask'][1,1]=False;o['range_m'][1,1]=np.nan
        r.add(o,np.zeros(3),o,0.,False,False)
        x=torch.from_numpy(canonical_image(o)).unsqueeze(0);y=r._batch(np.array([i]))['image']
        assert torch.equal(x,y) and torch.equal(radial_images(x),radial_images(y))
        module=NearBypass(LidarConfig().directions());assert torch.equal(module(x),module(y))
    p=tmp_path/'replay.pt';r.save(p);other=Replay.load(p,c)
    for name in ('ranges','masks','states','sensor_times'):
        np.testing.assert_array_equal(getattr(r,name),getattr(other,name))
    with pytest.raises(FileExistsError):r.save(p)
    with pytest.raises(ValueError):Replay.load(p,contract(PerceptionSpec()))
    bad=copy.deepcopy(o);bad['hit_mask'][1,1]=True
    before=r.total
    with pytest.raises(ValueError):r.add(bad,np.zeros(3),o,0.,False,False)
    assert r.total==before

def recursive_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(recursive_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b

@pytest.mark.parametrize('bypass',[False,True])
def test_td3_delay_target_update_and_exact_resume(tmp_path,bypass):
    spec=PerceptionSpec(bypass=bypass);c=contract(spec)
    cfg=copy.deepcopy(load_config()['td3']);cfg['batch_size']=4
    r=SuccessReplay(c,capacity=8,success_capacity=4)
    o=raw()
    for i in range(6):r.add(o,np.zeros(3,np.float32),o,.1,i==5,False)
    r.finish(True,10)
    agent=RangeTD3(cfg,c,spec);initial=copy.deepcopy(agent.state());torch.manual_seed(43)
    d1=agent.update(r);assert not d1['actor_updated'] and recursive_equal(initial['actor'],agent.state()['actor'])
    assert recursive_equal(initial['critic_target'],agent.state()['critic_target'])
    d2=agent.update(r);assert d2['actor_updated'] and not recursive_equal(initial['actor'],agent.state()['actor'])
    assert not recursive_equal(initial['critic_target'],agent.state()['critic_target'])
    assert agent.actor_updates==1 and r.last_sample_counts=={'main':4,'10':0}
    cp=agent.checkpoint();torch.save(cp,tmp_path/'agent.pt',pickle_protocol=4);r.save(tmp_path/'replay.pt')
    restored=RangeTD3(cfg,c,spec);restored.restore_checkpoint(torch.load(tmp_path/'agent.pt',weights_only=False))
    assert recursive_equal(agent.state(),restored.state())
    rr=SuccessReplay.load(tmp_path/'replay.pt',c);rng=torch.get_rng_state()
    a=agent.update(r);torch.set_rng_state(rng);b=restored.update(rr)
    assert a==b and recursive_equal(agent.state(),restored.state())
    wrong=RangeTD3(cfg,contract(replace(spec,bypass=not bypass)),replace(spec,bypass=not bypass))
    with pytest.raises(ValueError):wrong.restore_checkpoint(cp)
    old=copy.deepcopy(cp);old['format']='rv1_training';
    with pytest.raises(ValueError):restored.restore_checkpoint(old)

def test_ring_wrap_terminal_semantics_and_balancing(tmp_path):
    c=contract(PerceptionSpec());r=SuccessReplay(c,capacity=3,success_capacity=8,fraction=.5)
    o=raw()
    for d in (10,20):
        for i in range(3):r.add(o,np.zeros(3),o,float(d),i==1,i==2)
        r.finish(True,d)
    assert r.main.total==6 and r.main.pointer==0 and r.main.size==3
    p=tmp_path/'r.pt';r.save(p);b=SuccessReplay.load(p,c)
    rng=torch.get_rng_state();a=r.sample(8);torch.set_rng_state(rng);bb=b.sample(8)
    assert recursive_equal(a,bb) and r.last_sample_counts=={'main':4,'10':2,'20':2}
    fixed=r.main._batch(np.arange(3));assert fixed['state'].shape==(3,18)
    assert (~(r.main.terminated[:3]|r.main.truncated[:3])).ravel().tolist()==[True,False,False]
