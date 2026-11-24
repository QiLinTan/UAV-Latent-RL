import copy
import numpy as np
import pytest
import torch
from experiments.rv1_range_cnn.models import RangeEncoder
from experiments.rv1_range_cnn.provenance import contract,validate_contract
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.replay import Replay,SuccessReplay
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.checkpoint import capture,restore_environment,rng_state,restore_rng
from experiments.direct_velocity_full_v1.task_factory import load_config,make_env,ROOT
from envs.map_atlas.forest_pool import ForestPool

torch.set_num_threads(2)

@pytest.mark.parametrize('dimension',[32,64,128])
@pytest.mark.parametrize('shape',[(32,128),(16,256)])
def test_encoder_capacity_shapes_gradients(dimension,shape):
    enc=RangeEncoder(shape,dimension);image=torch.rand(4,3,*shape,requires_grad=True)
    z,spatial=enc(image,True);assert z.shape==(4,dimension) and spatial.shape==(4,32,shape[0]//8,shape[1]//8)
    (z.square().mean()+spatial.square().mean()).backward()
    assert torch.isfinite(image.grad).all() and all(torch.isfinite(p.grad).all() for p in enc.parameters())

def test_circular_seam_equivariance():
    enc=RangeEncoder();image=torch.rand(2,3,32,128)
    a=enc.convolutions(image);b=enc.convolutions(torch.roll(image,8,-1))
    torch.testing.assert_close(b,torch.roll(a,1,-1),atol=1e-6,rtol=1e-6)

def observation(seed=1):
    rng=np.random.default_rng(seed);r=rng.uniform(.1,8,(32,128)).astype(np.float32)
    return dict(state=rng.uniform(-1,1,18).astype(np.float32),range_m=r,hit_mask=np.ones((32,128),bool),scanned_mask=np.ones((32,128),bool))

def test_replay_td3_checkpoint_exact_next_update(tmp_path):
    torch.manual_seed(43);c=contract(LidarConfig(),'unit_test');cfg=copy.deepcopy(load_config()['td3']);cfg['batch_size']=8
    replay=SuccessReplay(c,capacity=32,success_capacity=16)
    for i in range(20):replay.add(observation(i),np.array([.2,.1,.1]),observation(i+1),.1,i==19,False,episode_id=0,timestamps=(i/15,(i+1)/15))
    replay.finish(True,10);agent=RangeTD3(cfg,c)
    before={k:p.detach().clone() for k,p in agent.actor.encoder.named_parameters()}
    for _ in range(4):
        d=agent.update(replay);assert np.isfinite(d['critic_loss'])
    assert any(not torch.equal(before[k],p) for k,p in agent.actor.encoder.named_parameters())
    assert agent.actor_updates==2 and replay.last_sample_counts['10']==1
    path=tmp_path/'replay.pt';replay.save(path);restored=SuccessReplay.load(path,c)
    ckpt=agent.checkpoint();second=RangeTD3(cfg,c);second.restore_checkpoint(ckpt)
    original_storage={v['exp_avg'].data_ptr() for v in agent.critic_optimizer.state.values() if 'exp_avg' in v}
    restored_storage={v['exp_avg'].data_ptr() for v in second.critic_optimizer.state.values() if 'exp_avg' in v}
    assert not original_storage & restored_storage
    rng=rng_state();a=agent.update(replay);restore_rng(rng);b=second.update(restored)
    assert a==b
    for x,y in zip(agent.actor.parameters(),second.actor.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    for x,y in zip(agent.critic.parameters(),second.critic.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    sample=replay.sample(8);assert sample[0]['image'].shape==(8,3,32,128)
    assert set(sample[0])=={'state','image'}
    with pytest.raises(ValueError):validate_contract({'schema':'dv0_114D_yaw_velocity_replay_v1'},c)
    with pytest.raises(ValueError):second.restore_checkpoint({'format':'dv0_training_v1','contract':c})
    before_total=replay.main.total
    with pytest.raises(ValueError):replay.add(observation(),[1,1,0],observation(),0,False,False)
    assert replay.main.total==before_total

def test_done_masks_and_unscanned_codec(tmp_path):
    c=contract(LidarConfig(),'unit_test');r=Replay(3,c);obs=observation();obs['scanned_mask'][0,0]=False;obs['hit_mask'][0,0]=False;obs['range_m'][0,0]=np.nan
    r.add(obs,[0,0,0],obs,1,False,True);_,_,_,_,mask=r.sample(7);assert not mask.any()
    p=tmp_path/'r.pt';r.save(p);copy=Replay.load(p,c);s=copy.sample(3)[0]['image']
    assert not s[:,:,0,0].any() and torch.isfinite(s).all()

def test_course_reset_and_exact_physical_resume():
    c=load_config();pool=ForestPool(ROOT/c['forest_pool_path']);new=make_course_env(pool);old=make_env(c,pool=pool)
    try:
        for seed in [43,44]:
            a,ai=new.reset(seed=seed);b,bi=old.reset(seed=seed)
            np.testing.assert_array_equal(a['state'],b.reshape(-1)[:18])
            for key in ('map_pool_index','density_stage','start_curriculum_source','start_curriculum_episode_level'):assert ai[key]==bi[key]
            np.testing.assert_allclose(new.base_env.pos,old.base_env.pos,atol=0)
            assert new.base_env.EPISODE_LEN_SEC==old.base_env.EPISODE_LEN_SEC
            assert new.lidar.scan_count==1 and new.controller.tick==0
        for _ in range(7):a,ra,ta,ua,ia=new.step([.1,-.1,0]);b,rb,tb,ub,ib=old.step([.1,-.1,0]);assert (ra,ta,ua)==(rb,tb,ub)
        saved=capture(new);clone=make_course_env(pool)
        try:
            restored=restore_environment(clone,saved)
            for key in a:np.testing.assert_array_equal(a[key],restored[key])
            x,rx,tx,ux,_=new.step([-.1,.1,.1]);y,ry,ty,uy,_=clone.step([-.1,.1,.1])
            assert (rx,tx,ux)==(ry,ty,uy)
            for key in x:np.testing.assert_array_equal(x[key],y[key])
        finally:clone.close()
    finally:new.close();old.close()
