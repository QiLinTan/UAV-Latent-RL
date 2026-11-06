import copy
import numpy as np
import pytest
import torch
from ..task_factory import load_config,make_env,data_contract,ROOT
from envs.map_atlas.forest_pool import ForestPool
from ..rl.versioned_replay import VersionedReplay
from ..rl.td3_adapter import TD3,project
from ..rl.checkpoint import capture_episode,restore_episode,validate,atomic_save


def setup():
    c=load_config();pool=ForestPool(ROOT/c['forest_pool_path']);return c,pool,data_contract(c,pool)


def test_schema_rejects_old_and_same_dimension_different_frame(tmp_path):
    c,pool,contract=setup();r=VersionedReplay(contract,100,40)
    for bad in ({'schema':'old917D'},dict(contract,action_spec={'frame':'world'})):
        with pytest.raises(ValueError):r.restore(dict(contract=bad))
        with pytest.raises(ValueError):validate(dict(format='dv0_training_v1',contract=bad),contract)
    with pytest.raises(ValueError):validate({'actor':torch.zeros(917)},contract)
    atomic_save(tmp_path/'checkpoint.pt',dict(format='dv0_training_v1',contract=contract))
    validate(torch.load(tmp_path/'checkpoint.pt',weights_only=False),contract)


def test_replay_mask_stratification_roundtrip_and_projected_td3():
    torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(43)
    c,pool,contract=setup();r=VersionedReplay(contract,1000,400)
    for density in (10,20,30,40):
        for i in range(80):
            s=np.random.uniform(-1,1,114).astype(np.float32)
            r.push(s,[.2,.1,0],.1,s+.01,i==79,[.16,.08,0],8)
        r.finish(True,density)
    states,actions,next_states,rewards,masks=r.sample(256)
    assert r.last_sample_counts=={'main':205,10:13,20:13,30:13,40:12}
    assert set(torch.unique(masks).tolist())<={0.,1.}
    fresh=VersionedReplay(contract,1000,400);fresh.restore(r.state())
    assert fresh.main.size==320 and all(b.size==80 for b in fresh.success.values())
    agent=TD3(c['td3']);before=[p.clone() for p in agent.actor.parameters()]
    qbefore=[p.clone() for p in agent.critic.parameters()]
    d=agent.update(r);assert not d['actor_updated']
    assert all(torch.equal(a,b) for a,b in zip(before,agent.actor.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(qbefore,agent.critic.parameters()))
    d=agent.update(r);assert d['actor_updated'] and np.isfinite(d['critic_loss'])
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.actor.parameters()))
    x=project(torch.tensor([[1.,1.,2.],[-2.,.2,-4.]]))
    assert torch.all(torch.linalg.vector_norm(x[:,:2],dim=1)<=1.000001)
    agent2=TD3(c['td3']);agent2.restore(agent.state())
    np.testing.assert_array_equal(agent.action(states[0].numpy()),agent2.action(states[0].numpy()))


def test_restore_episode_retains_control_reward_rng_and_next_transition():
    c,pool,contract=setup();env=make_env(pool=pool);second=make_env(pool=pool)
    try:
        obs,_=env.reset(seed=43)
        for i in range(5):obs,*_=env.step([-.2,.25,.05])
        saved=capture_episode(env)
        restored=restore_episode(second,saved)
        np.testing.assert_allclose(obs,restored,atol=2e-6)
        assert env.controller.snapshot()==second.controller.snapshot()
        for action in ([.2,.15,0],[0,0,-.1],[-.3,-.1,0]):
            a=env.step(action);b=second.step(action)
            np.testing.assert_allclose(a[0],b[0],atol=2e-5)
            assert a[1]==pytest.approx(b[1],abs=2e-5) and a[2:4]==b[2:4]
    finally:env.close();second.close()
